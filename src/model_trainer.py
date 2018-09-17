import itertools
import os
import shutil
import tensorflow as tf
import time
import operator
from tensorflow.python.client import device_lib
from tensorflow.python.training import device_setter
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2

def add_grads_summary(grads):
    with tf.name_scope('grads_summary'):
        for grad, var in grads:
            if grad is not None:
                grad_ = tf.boolean_mask(grad, tf.is_finite(grad))
                tf.summary.scalar('gradients/' + var.op.name, tf.norm(grad_))
                tf.summary.histogram('gradients/' + var.op.name, grad_)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
              '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()
        
    return _local_device_chooser


class ModelBuilder:
    def __init__(self):
        self._forward_fn = None
        self._loss_fn = None
        self._exclude_params = None

    def set_exclude_params(self, exclude_params):
        # callable or list or tuple or None
        self._exclude_params = exclude_params
        return self

    def set_forward(self, forward):
        if not callable(forward):
            raise ValueError('forward: is not callable')

        self._forward_fn = forward
        return self

    def set_loss(self, loss):
        if not callable(loss):
            raise ValueError('loss: is not callable')
            
        self._loss_fn = loss
        return self

    def build(self):
        forward_fn = self._forward_fn
        loss_fn = self._loss_fn
        exclude_params = self._exclude_params
        
        class AnonymousModel:
            def __init__(self):
                self._forward_fn = forward_fn
                self._loss_fn = loss_fn
                self._exclude_params = exclude_params

            def forward(self, is_training_mode, *inputs):
                self.inputs = inputs
                
                self.outputs = self._forward_fn(is_training_mode, *self.inputs)
                    
                params = tf.trainable_variables()

                if callable(self._exclude_params):
                    exclude_params = self._exclude_params()
                else:
                    exclude_params = self._exclude_params

                if exclude_params:
                    params = [param for param in params if param.name not in exclude_params]

                self.params = params

                return self.outputs

            def loss(self, scope=None):
                outputs = self.outputs
                if not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]
                    
                self._loss_fn(*self.inputs, *outputs)
                
                losses = tf.losses.get_losses(scope=scope)
                reg_losses = tf.losses.get_regularization_losses(scope=scope)

                self._loss = tf.add_n(losses)
                if len(reg_losses):
                    self._loss = train_loss + tf.add_n(reg_losses)
                
                self.losses = losses
                self.reg_losses = reg_losses

                return losses

            def gradients(self):
                _params = self.params

                grads = tf.gradients(self._loss, _params)
                return list(zip(grads, _params))
        
        return AnonymousModel


class Trainer:
    def __init__(self, n_training_steps=300000,
                       n_summary_steps=500,
                       n_dataset_workers=8,
                       batch_size=24,
                       buffer_size=100,
                       learning_rate=1e-4,
                       learning_rate_decay=0.99,
                       learning_rate_decay_staircase=False,
                       n_learning_rate_decay_steps=2000,
                       dropout_keep_prob=1,
                       grad_clip_value=20,
                       n_checkpoint_steps=2500,
                       gpu_memory_fraction=1.,
                       training_dir_path='./training',
                       allow_restoring=True,
                       dataset_enable_caching=False,
                       dataset_cache_dir_path=None,
                       place_vars_on_cpu=False,
                       use_gready_placement_startegy=False,
                       grads_sync_steps=1):
        self.n_training_steps = n_training_steps
        self.n_summary_steps = n_summary_steps
        self.n_dataset_workers = n_dataset_workers
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_staircase = learning_rate_decay_staircase
        self.n_learning_rate_decay_steps = n_learning_rate_decay_steps
        self.grad_clip_value = grad_clip_value
        self.n_checkpoint_steps = n_checkpoint_steps
        self.gpu_memory_fraction = gpu_memory_fraction
        self.allow_restoring = allow_restoring
        self.dropout_keep_prob = dropout_keep_prob

        self.training_dir_path = training_dir_path
        self.gpu_memory_fraction = gpu_memory_fraction

        self.dataset_enable_caching = dataset_enable_caching
        self.dataset_cache_dir_path = dataset_cache_dir_path
        self.dataset_needs_flatting = False

        self.place_vars_on_cpu = place_vars_on_cpu
        self.use_gready_placement_startegy = use_gready_placement_startegy
        
        self.grads_sync_steps = max(1, grads_sync_steps)

        self._is_builded = False
        self.saver = None

        self.datasets = []

    def add_inputs(self, dataset_placeholders_getter, dataset_mapper=None, needs_flatting=False):
        '''
        Arguments
        ---------
        dataset_placeholders_getter
        dataset_mapper
        needs_flatting
        '''
        if not callable(dataset_placeholders_getter):
            raise ValueError('dataset_placeholders_getter: is not callable')

        if dataset_mapper is not None and not callable(dataset_mapper):
            raise ValueError('dataset_mapper: is not callable')

        self.datasets.append((dataset_placeholders_getter, needs_flatting, dataset_mapper))
        
        self._is_builded = False

        return self
    
    def set_model(self, model_getter):
        '''
        Arguments
        ---------
        model_getter
        '''
        if not hasattr(model_getter, 'forward'):
            raise ValueError('model_getter: has not `forward` method')

        if not hasattr(model_getter, 'loss'):
            raise ValueError('model_getter: has not `loss` method')

        if not hasattr(model_getter, 'gradients'):
            raise ValueError('model_getter: has not `gradients` method')

        if isinstance(model_getter, type):
            self._model_getter = model_getter
        else:
            self._model_getter = lambda: model_getter

        self._is_builded = False

        return self
    
    def set_metrics(self, metrics_getter):
        '''
        Arguments
        ---------
        metrics_getter
        '''
        self.metrics_getter = metrics_getter

        self._is_builded = False

        return self

    def set_summary(self, summary_getter):
        '''
        Arguments
        ---------
        summary_getter
        '''
        self.summary_getter = summary_getter

        self._is_builded = False

        return self
    
    def train(self, train_data_sources, valid_data_sources, model_initial_weights_loader=None,
              verbose=False, training_dir_path=None, auto_freeze=None):
        self._build_graph()

        TRAINING_DIR = self.training_dir_path
        if training_dir_path:
            TRAINING_DIR = training_dir_path

        GPU_MEMORY_FRACTION = self.gpu_memory_fraction

        TRAINING_STEPS = self.n_training_steps
        STEPS_PER_CHECKPOINT = self.n_checkpoint_steps
        STEPS_PER_SUMMARY = self.n_summary_steps
        ALLOW_RESTORING = self.allow_restoring

        if not os.path.exists(TRAINING_DIR):
            os.makedirs(TRAINING_DIR)

        checkpoint_path = os.path.join(TRAINING_DIR, 'model.ckpt')

        def setup_feed(data_sources, feed_dict):
            if callable(data_sources):
                feed_dict.update(data_sources(*self.datasets_placeholders))
            elif isinstance(data_sources, (tuple, list)):
                assert len(self.datasets_placeholders) == len(data_sources), 'Incorrect number of sources'
                for placeholders, data_source in zip(self.datasets_placeholders, data_sources):
                    feed_dict.update(data_source(*placeholders))
            else:
                raise ValueError('Unknown data source format')

        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            try:
                if verbose:
                    print('Initializing parameters ', flush=True, end='')
                
                sess.run(self._init_globals_op)
                sess.run(self._init_locals_op)

                train_iter_feed_dict = dict()
                setup_feed(train_data_sources, train_iter_feed_dict)
                if self.pipe_name_tf_phr is not None:
                    train_iter_feed_dict[self.pipe_name_tf_phr] = 'train'

                sess.run(self.train_iterator.initializer, train_iter_feed_dict)

                valid_iter_feed_dict = dict()
                setup_feed(valid_data_sources, valid_iter_feed_dict)
                if self.pipe_name_tf_phr is not None:
                    valid_iter_feed_dict[self.pipe_name_tf_phr] = 'valid'

                sess.run(self.valid_iterator.initializer, valid_iter_feed_dict)
            
                if verbose:
                    print('[OK]', flush=True)
            except:
                if verbose:
                    print('[Failed]', flush=True) 
                raise

            if model_initial_weights_loader is not None:
                model_initial_weights_loader(sess)
            else:
                model = self._towers_models[0]
                if hasattr(model, 'preload_weights_op') and callable(model.preload_weights_op):
                    model.preload_weights_op()(sess)
                    
            ckpt = tf.train.get_checkpoint_state(TRAINING_DIR)
            if ALLOW_RESTORING and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                shutil.rmtree(TRAINING_DIR)
                self.saver.save(sess, checkpoint_path)

            tf.train.write_graph(sess.graph_def, TRAINING_DIR, 'graph.pb', as_text=False)
            
            _train_summary_writer = tf.summary.FileWriter(os.path.join(TRAINING_DIR, 'summary', 'train'), sess.graph)
            _valid_summary_writer = tf.summary.FileWriter(os.path.join(TRAINING_DIR, 'summary', 'valid'), sess.graph)
            
            try:
                _step = int(sess.run(self._step_var))
                if _step == 0:
                    _train_loss, _train_summary = sess.run([self.total_loss, self.train_summary_op], {self.data_loader_mode: 'train-pipe'})
                    _valid_loss, _valid_summary = sess.run([self.total_loss, self.valid_summary_op], {self.data_loader_mode: 'valid-pipe'})
                    _train_summary_writer.add_summary(_train_summary, _step)
                    _valid_summary_writer.add_summary(_valid_summary, _step)

                    if verbose:
                        print('Initial train loss = %.6f, valid loss = %.6f.' % (_train_loss, _valid_loss), flush=True)

                if verbose:
                    print('Start trainging.', flush=True)
                
                if verbose:
                    start = time.time()
                    
                for _ in range(_step, TRAINING_STEPS):
                    run_metadata = tf.RunMetadata()
                    sess.run(self.train_op, {self.is_training_mode: True, self.data_loader_mode: 'train-pipe'}, run_metadata=run_metadata)
                    
                    _step = int(sess.run(self._step_var))
                    
                    if _step % STEPS_PER_SUMMARY == 0:
                        _train_loss, _train_summary = sess.run([self.total_loss, self.train_summary_op], {self.data_loader_mode: 'train-pipe'})
                        _valid_loss, _valid_summary = sess.run([self.total_loss, self.valid_summary_op], {self.data_loader_mode: 'valid-pipe'})
                        _train_summary_writer.add_summary(_train_summary, _step)
                        _train_summary_writer.add_run_metadata(run_metadata, 'train-op-%i' % _step, _step)
                        _valid_summary_writer.add_summary(_valid_summary, _step)
                        
                        if verbose:
                            elapsed = time.time() - start
                            start = time.time()
                            print('Step #%i: train loss = %.6f, valid loss = %.6f, elapsed %.3f sec.' % (_step, _train_loss, _valid_loss, elapsed), flush=True)

                    if _step % STEPS_PER_CHECKPOINT == 0:
                        self.saver.save(sess, checkpoint_path, global_step=_step)

                if verbose:
                    print('Training process is finished.', flush=True)
            finally:
                self.saver.save(sess, checkpoint_path, global_step=_step)
                tf.train.write_graph(sess.graph_def, TRAINING_DIR, 'graph.pb', as_text=False)
                if auto_freeze:
                    try:
                        if verbose:
                            print('Freezing...',  flush=True, end='')

                        self.freeze(**auto_freeze)
                        
                        if verbose:
                            print('[OK]', flush=True)
                    except:
                        if verbose:
                            print('[Failed]', flush=True) 
                        raise

    def freeze(self, input_getter, outputs_names=None, training_dir_path=None, ckpt_path=None, graph_protected_nodes=None):
        if not input_getter:
            raise ValueError('input_getter: is empty')
            
        if not callable(input_getter):
            raise ValueError('input_getter: is not callable')
            
        if training_dir_path is None:
            training_dir_path = self.training_dir_path
            
        ckpt = tf.train.get_checkpoint_state(training_dir_path)
        
        if ckpt_path is None:
            ckpt_path = ckpt.model_checkpoint_path if ckpt else None
        
        if not ckpt_path or not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            raise ValueError('Model is not trained.')
            
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope('model') as scope:
                inputs = input_getter()
                model = self._model_getter()
                if hasattr(model, 'inference') and callable(model.inference):
                    outputs = model.inference(*inputs)
                else:
                    outputs = model.forward(False, *inputs)
                    
                if not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]
                    
                tf.graph_util.remove_training_nodes(graph.as_graph_def(), graph_protected_nodes)
                
                model_saver = tf.train.Saver(tf.trainable_variables())
                
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)), graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            if outputs_names:
                outputs = [sess.graph.get_tensor_by_name(item) for item in outputs_names]

            model_saver.restore(sess, ckpt.model_checkpoint_path)
            
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                [node.name.split(':')[0] for node in outputs]
            )

            with tf.gfile.GFile(os.path.join(training_dir_path, 'graph.frozen.pb'), "wb") as f:
                f.write(output_graph_def.SerializeToString())

            print('%d ops in the final graph.' % len(output_graph_def.node))
            print('The frozen graph is stored in file: `%s`' % os.path.join(training_dir_path, 'graph.frozen.pb'))
    
    #
    # private section
    #

    def _build_graph(self):
        if not self._is_builded:
            tf.reset_default_graph()

            self._gpus = get_available_gpus()

            def build():
                with tf.name_scope('dataset'):
                    self._setup_dataset()

                with tf.name_scope('model') as scope:
                    with tf.name_scope('placeholders'):
                        self.is_training_mode = tf.placeholder_with_default(False, [], name='is_training_mode')
                        self.data_loader_mode = tf.placeholder_with_default('train-pipe', [], name='data_loader_mode')

                    self._setup_model(scope)

                with tf.name_scope('training'):
                    self._setup_train_op()
                    
                    self.saver = tf.train.Saver(tf.global_variables())

                    self._setup_metrics()
                    
                self._setup_summary()
                
                self._init_globals_op = tf.global_variables_initializer()
                self._init_locals_op = tf.local_variables_initializer()

                self._is_builded = True

            if self.place_vars_on_cpu:
                with tf.device('/cpu:0'):
                    return build()
            elif self._gpus:
                with tf.device(self._gpus[0]):
                    return build()
            else:
                return build()

    def _setup_dataset(self):
        N_TRAINERS = max(1, len(self._gpus))

        if self.dataset_cache_dir_path:
            CACHE_DIR_PATH = self.dataset_cache_dir_path if self.dataset_cache_dir_path.endswith('/') else (self.dataset_cache_dir_path + '/')
        else:
            CACHE_DIR_PATH = None

        datasets = []
        datasets_placeholders = []
        for i, (dataset_placeholders_getter, dataset_needs_flatting, dataset_mapper) in enumerate(self.datasets):
            dataset_placeholders = dataset_placeholders_getter()
            if not isinstance(dataset_placeholders, (tuple, list)):
                raise ValueError('dataset_placeholders: is neither a tuple nor a list')

            if self.dataset_enable_caching and CACHE_DIR_PATH is not None:
                pipe_name_tf_phr = tf.placeholder(tf.string, name='pipe_name')
            else:
                pipe_name_tf_phr = None

            dataset = tf.data.Dataset().from_tensor_slices(dataset_placeholders)

            if dataset_mapper is not None:
                dataset = dataset.map(dataset_mapper, self.n_dataset_workers)
                dataset = dataset.apply(tf.contrib.data.ignore_errors())

            if self.dataset_enable_caching:
                if CACHE_DIR_PATH is not None and pipe_name_tf_phr is not None:
                    if not os.path.exists(CACHE_DIR_PATH):
                        os.makedirs(CACHE_DIR_PATH)
                    dataset = dataset.cache(tf.constant(CACHE_DIR_PATH + ('data-%i-' % i)) + pipe_name_tf_phr)
                else:
                    dataset = dataset.cache()

            if dataset_needs_flatting:
                dataset = dataset.flat_map(lambda *samples: tf.data.Dataset.from_tensor_slices(samples))

            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            dataset = dataset.repeat()

            if len(datasets) > 0:
                assert datasets[-1].output_shapes == dataset.output_shapes and datasets[-1].output_types == dataset.output_types,\
                    'Datasets don\'t produce the same types of elements' 

            datasets.append(dataset)
            datasets_placeholders.append(dataset_placeholders)

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = tf.data.Dataset.from_tensor_slices(datasets)
            dataset = dataset.interleave(lambda d: d, cycle_length=len(datasets), block_length=1)

        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=N_TRAINERS*(self.grads_sync_steps if N_TRAINERS > 1 else 1))

        self.pipe_name_tf_phr = pipe_name_tf_phr

        self.datasets_placeholders = datasets_placeholders
        self.dataset = dataset

        self.train_iterator = dataset.make_initializable_iterator('train')
        self.valid_iterator = dataset.make_initializable_iterator('valid')
        
        self.train_batch = self.train_iterator.get_next
        self.valid_batch = self.valid_iterator.get_next

    def _get_device_setter(self, name):
        if self.place_vars_on_cpu:
            return local_device_setter(worker_device=name)
        else:
            return local_device_setter(
                ps_device_type='gpu',
                worker_device=name,
                ps_strategy=(None if not self.use_gready_placement_startegy
                             else tf.contrib.training.GreedyLoadBalancingStrategy(
                                len(self._gpus), tf.contrib.training.byte_size_load_fn)))

    def _setup_model(self, parent_scope):
        self._towers_models = []
        self._towers_grads = []

        def build_model(scope, grads_factor=1.):
            _batch = tf.case([(tf.equal(self.data_loader_mode, 'train-pipe'), lambda: self.train_batch()),
                              (tf.equal(self.data_loader_mode, 'valid-pipe'), lambda: self.valid_batch())],
                             exclusive=True)

            _batch = [tf.identity(item, name='batch/item-%i' % i) for i, item in enumerate(_batch)]
            
            model = self._model_getter()
            with scope as scope:
                model.forward(self.is_training_mode, *_batch)

                losses = model.loss(scope)
                
                if not losses:
                    losses = tf.losses.get_losses(scope=scope)
                    
                gradvars = model.gradients()
                
                if self.grad_clip_value is not None:
                    with tf.name_scope('grads-clipping'):
                        grad_clip_value = tf.constant(self.grad_clip_value, dtype=tf.float32)
                        gradvars = [((tf.clip_by_norm(grad, grad_clip_value) if grad is not None else grad), var) for grad, var in gradvars]

                if len(self._gpus) > 1:
                    with tf.name_scope('grads-division-for-avg'):
                        multiplier = tf.constant(grads_factor / len(self._gpus), dtype=tf.float32)
                        gradvars = [((tf.multiply(grad, multiplier) if grad is not None else grad), var) for grad, var in gradvars]

                return model, losses, gradvars

        var_scope = tf.get_variable_scope()
        if len(self._gpus) > 1:
            for i, name in enumerate(self._gpus):
                with tf.device(self._get_device_setter(name)):
                    result_set = []
                    for s in range(self.grads_sync_steps):
                        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
                            if self.grads_sync_steps == 1:
                                scope = tf.name_scope('tower-%i' % i)
                            else:
                                scope = tf.name_scope('tower-%i-%i' % (i, s))

                            result = build_model(scope, grads_factor=1./self.grads_sync_steps)

                            result_set.append(result)

                    if len(result_set) == 1:
                        model, losses_, gradvars = result_set[0]
                    else:
                        models, losses_list, gradvars = zip(*result_set)

                        model = models[0]
                        losses_ = losses_list[0]

                        with tf.name_scope('gradient-summing'):
                            all_grads = {}
                            for grad, var in itertools.chain(*gradvars):
                                if grad is not None:
                                    all_grads.setdefault(var, []).append(grad)

                            gradvars = []
                            for var, grads in all_grads.items():
                                if len(grads) == 1:
                                    avg_grad = grads[0]
                                else:
                                    avg_grad = tf.add_n(grads)
                                
                                with tf.device(var.device):
                                    avg_grad = tf.identity(avg_grad)
                                        
                                gradvars.append((avg_grad, var))

                    self._towers_models.append(model)
                    self._towers_grads.append(gradvars)

                    if i == 0:
                        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=parent_scope)
                        losses = losses_
        else:
            with tf.variable_scope(var_scope):
                if len(self._gpus):
                    with tf.device(self._get_device_setter(self._gpus[0])):
                        model, losses, gradvars = build_model(tf.name_scope(parent_scope))
                else:
                    model, losses, gradvars = build_model(tf.name_scope(parent_scope))
                
                self._towers_models.append(model)
                self._towers_grads.append(gradvars)
                
                self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=parent_scope)

        self.total_loss = tf.add_n(losses)

    def _setup_train_op(self):
        step_var = tf.Variable(0, trainable=False)

        with tf.name_scope('optimizer'):
            with tf.name_scope('params'):
                lr_var = tf.Variable(self.learning_rate, trainable=False)
                    
                if self.learning_rate_decay and self.n_learning_rate_decay_steps:
                    lr_var = tf.train.exponential_decay(
                        lr_var, step_var, self.n_learning_rate_decay_steps, self.learning_rate_decay,
                        staircase=self.learning_rate_decay_staircase)

            if len(self._gpus) > 1:
                n_steps = self.grads_sync_steps
            else:
                n_steps = 1
                
            self._learning_rate = lr_var * n_steps

            _optimizer = tf.train.AdamOptimizer(lr_var)
        
            with tf.name_scope('gradient-summing'):
                all_grads = {}
                for grad, var in itertools.chain(*self._towers_grads):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                        
                gradvars = []
                for var, grads in all_grads.items():
                    with tf.device(var.device):
                        if len(grads) == 1:
                            avg_grad = tf.identity(grads[0])
                        else:
                            avg_grad = tf.add_n(grads)
                    gradvars.append((avg_grad, var))
            
            apply_gradient_op = _optimizer.apply_gradients(gradvars, global_step=step_var)

            self._grads = gradvars
            self.train_op = apply_gradient_op
            self._step_var = step_var * n_steps

            if self._update_ops:         
                self.train_op = tf.group(self.train_op, *self._update_ops)

    def _setup_metrics(self):
        with tf.name_scope('metrics'):
            model = self._towers_models[0]
            with tf.device(self._get_device_setter(self._gpus[0]) if len(self._gpus) > 1 else None):
                if hasattr(model, 'metrics') and callable(model, 'metrics'):
                    metrics = model.metrics()
                else:
                    metrics = self.metrics_getter(model)

            self._metrics = metrics

    def _setup_summary(self):
        model = self._towers_models[0]
        if hasattr(model, 'summary') and callable(model, 'summary'):
            ops = model.summary(self._step_var, self._learning_rate, self._grads)
        else:
            ops = self.summary_getter(
                model, self._step_var, self._learning_rate, self._grads, self._metrics)
        self.train_summary_op, self.valid_summary_op = ops

        