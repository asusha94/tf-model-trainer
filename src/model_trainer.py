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
        self._model_fn = None
        self._loss_fn = None
        self._exclude_params = None

    def set_exclude_params(self, exclude_params):
        # callable or list or tuple or None
        self._exclude_params = exclude_params
        return self

    def set_model(self, model):
        if not callable(model):
            raise ValueError('model: is not callable')

        self._model_fn = model
        return self

    def set_loss(self, loss):
        if not callable(loss):
            raise ValueError('loss: is not callable')
            
        self._loss_fn = loss
        return self

    def __call__(self):
        class AnonymousModel:
            def __init__(self, model_fn, loss_fn, exclude_params):
                self._model_fn = model_fn
                self._loss_fn = loss_fn
                self._exclude_params = exclude_params

            def forward(self, is_training_mode, *inputs):
                self.inputs = inputs
                
                self.outputs = self._model_fn(is_training_mode, *self.inputs)
                if not isinstance(self.outputs, (tuple, list)):
                    self.outputs = [self.outputs]
                    
                params = tf.trainable_variables()

                if callable(self._exclude_params):
                    exclude_params = self._exclude_params()
                else:
                    exclude_params = self._exclude_params

                if exclude_params:
                    params = [param for param in params if param.name not in exclude_params]

                self._params = params

                return self.outputs

            def loss(self, scope=None):
                self._loss_fn(*self.inputs, *self.outputs)
                
                losses = tf.losses.get_losses(scope=scope)
                reg_losses = tf.losses.get_regularization_losses(scope=scope)

                loss = tf.add_n(losses)
                if len(reg_losses):
                    loss = loss + tf.add_n(reg_losses)

                self._loss = loss

                return losses, reg_losses

            def gradients(self):
                _params = self._params

                grads = tf.gradients(self._loss, _params)
                return list(zip(grads, _params))
        
        return AnonymousModel(self._model_fn, self._loss_fn, self._exclude_params)


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
    
    def set_model(self, model):
        '''
        Arguments
        ---------
        model
        '''
        if callable(model):
            model = model()

        if not hasattr(model, 'forward'):
            raise ValueError('model: has not `forward` method')

        if not hasattr(model, 'loss'):
            raise ValueError('model: has not `loss` method')

        if not hasattr(model, 'gradients'):
            raise ValueError('model: has not `gradients` method')

        self.model = model

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
              pre_start_hooks=[], pre_train_hooks=[], post_train_hooks=[], pre_end_hooks=[], verbose=False, training_dir_path=None):
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

            if pre_start_hooks:
                for item in pre_start_hooks:
                    if callable(item):
                        item(sess)

            if model_initial_weights_loader is not None:
                model_initial_weights_loader(sess)
                    
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
                    if pre_train_hooks:
                        for item in pre_train_hooks:
                            if callable(item):
                                item(sess, _step)

                    run_metadata = tf.RunMetadata()
                    sess.run(self.train_op, {self.is_training_mode: True, self.data_loader_mode: 'train-pipe'}, run_metadata=run_metadata)
                    
                    _step = int(sess.run(self._step_var))

                    if post_train_hooks:
                        for item in post_train_hooks:
                            if callable(item):
                                item(sess, _step)
                    
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
        self._towers_outputs = []
        self._towers_losses = []
        self._towers_grads = []

        def build_model(scope, grads_factor=1.):
            _batch = tf.case([(tf.equal(self.data_loader_mode, 'train-pipe'), lambda: self.train_batch()),
                              (tf.equal(self.data_loader_mode, 'valid-pipe'), lambda: self.valid_batch())],
                             exclusive=True)

            _batch = [tf.identity(item, name='batch/item-%i' % i) for i, item in enumerate(_batch)]

            with scope as scope:
                outputs = self.model.forward(self.is_training_mode, *_batch)
                if not isinstance(outputs, (tuple, list)):
                    outputs = [outputs]

                losses, reg_losses = self.model.loss(scope)
                    
                gradvars = self.model.gradients()
                
                if self.grad_clip_value is not None:
                    with tf.name_scope('grads-clipping'):
                        grad_clip_value = tf.constant(self.grad_clip_value, dtype=tf.float32)
                        gradvars = [((tf.clip_by_norm(grad, grad_clip_value) if grad is not None else grad), var) for grad, var in gradvars]

                if len(self._gpus) > 1:
                    with tf.name_scope('grads-division-for-avg'):
                        multiplier = tf.constant(grads_factor / len(self._gpus), dtype=tf.float32)
                        gradvars = [((tf.multiply(grad, multiplier) if grad is not None else grad), var) for grad, var in gradvars]

                return (outputs, _batch), (losses, reg_losses), gradvars

        if len(self._gpus) > 1:
            for i, name in enumerate(self._gpus):
                with tf.device(self._get_device_setter(name)):
                    result_set = []
                    for s in range(self.grads_sync_steps):
                        with tf.variable_scope(parent_scope, reuse=tf.AUTO_REUSE):
                            if self.grads_sync_steps == 1:
                                scope = tf.name_scope('tower-%i' % i)
                            else:
                                scope = tf.name_scope('tower-%i-%i' % (i, s))

                            result = build_model(scope, grads_factor=1./self.grads_sync_steps)

                            result_set.append(result)

                    if len(result_set) == 1:
                        (outputs, _batch), (losses, reg_losses), gradvars = result_set[0]
                    else:
                        data, losses, gradvars = zip(*result_set)

                        outputs, _batch = data[0]
                        losses, reg_losses = losses[0]

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

                    self._towers_outputs.append((outputs, _batch))
                    self._towers_losses.append((losses, reg_losses))

                    self._towers_grads.append(gradvars)

                    if i == 0:
                        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=parent_scope)
                        self._losses = losses
        else:
            with tf.variable_scope(parent_scope):
                if len(self._gpus):
                    with tf.device(self._get_device_setter(self._gpus[0])):
                        (outputs, _batch), (losses, reg_losses), gradvars = build_model(tf.name_scope(parent_scope))
                else:
                    (outputs, _batch), (losses, reg_losses), gradvars = build_model(tf.name_scope(parent_scope))
                
                self._towers_outputs.append((outputs, _batch))
                self._towers_losses.append((losses, reg_losses))

                self._towers_grads.append(gradvars)
                
                self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=parent_scope)
                self._losses = losses

        self.total_loss = tf.add_n(self._losses)

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
            outputs, batch = self._towers_outputs[0]
            with tf.device(self._get_device_setter(self._gpus[0]) if len(self._gpus) > 1 else None):
                metrics = self.metrics_getter(*batch, *outputs)

            self._metrics = metrics

    def _setup_summary(self):
        ops = self.summary_getter(
            self._step_var, self._learning_rate, self._grads, self.total_loss, self._losses, self._metrics)
        self.train_summary_op, self.valid_summary_op = ops

        