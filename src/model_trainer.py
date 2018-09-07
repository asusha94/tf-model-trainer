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
                       place_vars_on_cpu=False):
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

        self._is_builded = False
        self.saver = None

        self.datasets = []

    def add_inputs(self, dataset_placeholders_getter, dataset_mapper=None, flat_mapper=False):
        '''
        Arguments
        ---------
        dataset_placeholders_getter
        dataset_mapper
        '''
        if not callable(dataset_placeholders_getter):
            raise ValueError('dataset_placeholders_getter: is not callable')

        # if not isinstance(output_dtypes, (tuple, list)):
        #     raise ValueError('output_dtypes: is neither a tuple nor a list')

        # if not isinstance(output_shapes, (tuple, list)):
        #     raise ValueError('output_shapes: is neither a tuple nor a list')

        # assert len(output_dtypes) == len(output_shapes), '%i == %i' % (len(output_dtypes), len(output_shapes))

        if dataset_mapper is not None and not callable(dataset_mapper):
            raise ValueError('dataset_mapper: is not callable')

        self.datasets.append((dataset_placeholders_getter, flat_mapper, dataset_mapper))
        
        self._is_builded = False

        return self
    
    def set_model(self, model_getter, model_exclude_params=None):
        self.model_getter = model_getter
        self.model_exclude_params = model_exclude_params

        self._is_builded = False

        return self
    
    def set_loss(self, loss_getter):
        self.loss_getter = loss_getter

        self._is_builded = False

        return self
    
    def set_metrics(self, metrics_getter):
        self.metrics_getter = metrics_getter

        self._is_builded = False

        return self

    def set_summary(self, summary_getter):
        self.summary_getter = summary_getter

        self._is_builded = False

        return self
    
    def train(self, train_data_sources, valid_data_sources, model_initial_weights_loader=None, pre_start_hooks=[], pre_train_hooks=[], post_train_hooks=[], pre_end_hooks=[], verbose=False, training_dir_path=None):
        self._build_graph()

        TRAINING_DIR = self.training_dir_path
        if training_dir_path:
            TRAINING_DIR = training_dir_path

        GPU_MEMORY_FRACTION = self.gpu_memory_fraction

        TRAINING_STEPS = self.n_training_steps
        STEPS_PER_CHECKPOINT = self.n_checkpoint_steps
        STEPS_PER_SUMMARY = self.n_summary_steps
        DROPOUT_KEEP_PROB = self.dropout_keep_prob
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
                    _train_loss, _train_summary = sess.run([self.loss, self.train_summary_op], {self.data_loader_mode: 'train-pipe'})
                    _valid_loss, _valid_summary = sess.run([self.loss, self.valid_summary_op], {self.data_loader_mode: 'valid-pipe'})
                    _train_summary_writer.add_summary(_train_summary, _step)
                    _valid_summary_writer.add_summary(_valid_summary, _step)
                    
                    _step = int(sess.run(self._step_inc_op))

                    if verbose:
                        print('Initial train loss = %.6f, valid loss = %.6f.' % (_train_loss, _valid_loss), flush=True)

                if verbose:
                    print('Start trainging.', flush=True)
                
                if verbose:
                    start = time.time()
                    
                if verbose:
                    start_10 = time.time()
                for _ in range(_step, TRAINING_STEPS):
                    _step = int(sess.run(self._step_var))

                    if pre_train_hooks:
                        for item in pre_train_hooks:
                            if callable(item):
                                item(sess, _step)

                    run_metadata = tf.RunMetadata()
                    sess.run([self.train_op], {self.is_training_mode: True, self.data_loader_mode: 'train-pipe'}, run_metadata=run_metadata)
                    
                    if _step % 10 == 0:
                        elapsed = time.time() - start_10
                        start_10 = time.time()
                        print('Step #%i: elapsed %.3f sec.' % (_step, elapsed), flush=True)

                    if post_train_hooks:
                        for item in post_train_hooks:
                            if callable(item):
                                item(sess, _step)
                    
                    if _step % STEPS_PER_SUMMARY == 0:
                        _train_loss, _train_summary = sess.run([self.loss, self.train_summary_op], {self.data_loader_mode: 'train-pipe'})
                        _valid_loss, _valid_summary = sess.run([self.loss, self.valid_summary_op], {self.data_loader_mode: 'valid-pipe'})
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

    def _build_graph(self):
        if not self._is_builded:
            tf.reset_default_graph()

            self._gpus = get_available_gpus()

            with tf.name_scope('dataset'):
                self._setup_dataset()

            with tf.name_scope('model'):
                self._setup_model()

            with tf.name_scope('training'):
                self.summaries = []

                self._setup_loss()

                self._setup_train_op()

                self._setup_metrics()

                self.saver = tf.train.Saver(tf.global_variables())

            self._setup_summary()
            
            self._init_globals_op = self._place_on_good_device(lambda: tf.global_variables_initializer())
            self._init_locals_op = self._place_on_good_device(lambda: tf.local_variables_initializer())

            self._is_builded = True

    def _setup_dataset(self):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.buffer_size
        DATASET_N_WORKERS = self.n_dataset_workers
        ENABLE_CACHING = self.dataset_enable_caching

        N_TRAINERS = max(1, len(self._gpus))

        if self.dataset_cache_dir_path:
            CACHE_DIR_PATH = self.dataset_cache_dir_path if self.dataset_cache_dir_path.endswith('/') else (self.dataset_cache_dir_path + '/')
        else:
            CACHE_DIR_PATH = None

        def dataset_create():
            datasets = []
            datasets_placeholders = []
            for i, (dataset_placeholders_getter, dataset_needs_flatting, dataset_mapper) in enumerate(self.datasets):
                dataset_placeholders = dataset_placeholders_getter()
                if not isinstance(dataset_placeholders, (tuple, list)):
                    raise ValueError('dataset_placeholders: is neither a tuple nor a list')

                if ENABLE_CACHING and CACHE_DIR_PATH is not None:
                    pipe_name_tf_phr = tf.placeholder(tf.string, name='pipe_name')
                else:
                    pipe_name_tf_phr = None

                dataset = tf.data.Dataset().from_tensor_slices(dataset_placeholders)

                if dataset_mapper is not None:
                    dataset = dataset.map(dataset_mapper, DATASET_N_WORKERS)
                    dataset = dataset.apply(tf.contrib.data.ignore_errors())

                if ENABLE_CACHING:
                    if CACHE_DIR_PATH is not None and pipe_name_tf_phr is not None:
                        if not os.path.exists(CACHE_DIR_PATH):
                            os.makedirs(CACHE_DIR_PATH)
                        dataset = dataset.cache(tf.constant(CACHE_DIR_PATH + ('data-%i-' % i)) + pipe_name_tf_phr)
                    else:
                        dataset = dataset.cache()

                if dataset_needs_flatting:
                    dataset = dataset.flat_map(lambda *samples: tf.data.Dataset.from_tensor_slices(samples))

                # dataset = dataset.map(lambda *sample: tuple(tf.reshape(item, shape) for item, shape in zip(sample, self.dataset_output_shapes)), DATASET_N_WORKERS)
                dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
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

            dataset = dataset.batch(batch_size=BATCH_SIZE)
            dataset = dataset.prefetch(buffer_size=N_TRAINERS)

            self.pipe_name_tf_phr = pipe_name_tf_phr

            self.datasets_placeholders = datasets_placeholders
            self.dataset = dataset

            self.train_iterator = dataset.make_initializable_iterator('train')
            self.valid_iterator = dataset.make_initializable_iterator('valid')
            
            self.train_batch = self.train_iterator.get_next
            self.valid_batch = self.valid_iterator.get_next
        
        self._place_on_good_device(dataset_create)            

    def _setup_model(self):
        def setup_aux():
            with tf.name_scope('aux'):
                is_training_mode = tf.placeholder_with_default(False, [], name='is_training_mode')
                data_loader_mode = tf.placeholder_with_default('train-pipe', [], name='data_loader_mode')
                return is_training_mode, data_loader_mode
            
        is_training_mode, data_loader_mode = self._place_on_good_device(setup_aux)
        
        self._towers_losses = []
        avg_losses = []
        self._towers_outputs = []

        towers_grads = []
        if len(self._gpus) > 0:
            with tf.variable_scope(tf.get_variable_scope()):
                for i, name in enumerate(self._gpus):
                    if self.place_vars_on_cpu:
                        device_setter = local_device_setter(
                            worker_device=name)
                    else:
                        device_setter = name
                        #device_setter = local_device_setter(
                        #    ps_device_type='gpu',
                        #    worker_device=name)#,
                            #ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                            #    len(self._gpus), tf.contrib.training.byte_size_load_fn))
                    with tf.device(device_setter):
                        def get_batch():
                            with tf.name_scope('inputs-batch-%i' % i):
                                _batch = tf.case([(tf.equal(data_loader_mode, 'train-pipe'), lambda: self.train_batch()),
                                                  (tf.equal(data_loader_mode, 'valid-pipe'), lambda: self.valid_batch())],
                                                 exclusive=True)
                                return _batch

                        _batch = get_batch() # self._place_on_good_device(get_batch)
            
                        with tf.name_scope('tower-%i' % i) as scope:
                            outputs = self.model_getter(True, *_batch)
                            self._towers_outputs.append((name, outputs, _batch))

                            _ = self.loss_getter(*_batch, *outputs)
                            losses = tf.losses.get_losses(scope)
                            reg_losses = tf.losses.get_regularization_losses(scope)
                            self._towers_losses.append((name, losses, reg_losses))
                            avg_losses.append(losses)
                            
                            loss = tf.add_n(losses)
                            if len(reg_losses):
                                loss = loss + tf.add_n(reg_losses)

                            _params = tf.trainable_variables()
        
                            if callable(self.model_exclude_params):
                                _excludes = self.model_exclude_params()
                                if _excludes:
                                    _params = list(filter(lambda x: any([item not in x.name for item in _excludes]), _params))

                            tower_grad = tf.gradients(loss, _params)
                            if len(self._gpus) > 1:
                                tower_grad = [(tf.multiply(grad, 1. / len(self._gpus)) if grad is not None else grad) for grad in tower_grad]
                                
                            tower_gradvars = list(zip(tower_grad, _params))
                            
                            if self.grad_clip_value is not None:
                                tower_gradvars = [(tf.clip_by_norm(grad, self.grad_clip_value), var) for grad, var in tower_gradvars]

                            towers_grads.append(tower_gradvars)

                            tf.get_variable_scope().reuse_variables()
        else:
            def get_batch():
                with tf.name_scope('inputs-batch'):
                    _batch = tf.case([(tf.equal(data_loader_mode, 'train-pipe'), lambda: self.train_batch()),
                                      (tf.equal(data_loader_mode, 'valid-pipe'), lambda: self.valid_batch())],
                                     exclusive=True)
                    return _batch

            _batch = get_batch() #self._place_on_good_device(get_batch, True)
            
            outputs = self.model_getter(True, *_batch)
            self._towers_outputs.append((None, outputs, _batch))

            _ = self.loss_getter(*_batch, *outputs)
            losses = tf.losses.get_losses()
            reg_losses = tf.losses.get_regularization_losses()
            self._towers_losses.append((None, losses, reg_losses))
            avg_losses.append(losses)

            loss = tf.add_n(losses)
            if len(reg_losses):
                loss = loss + tf.add_n(reg_losses)

            _params = tf.trainable_variables()
        
            if callable(self.model_exclude_params):
                _excludes = self.model_exclude_params()
                if _excludes:
                    _params = list(filter(lambda x: any([item not in x.name for item in _excludes]), _params))

            tower_grad = tf.gradients(loss, _params)
            towers_grads.append(zip(tower_grad, _params))

        self.is_training_mode = is_training_mode
        self.data_loader_mode = data_loader_mode

        self.towers_grads = towers_grads

        def calc_loss():
            self._avg_losses = [tf.reduce_mean(l, axis=0) for l in zip(*avg_losses)]
            self.loss = tf.add_n(self._avg_losses)

        self._place_on_good_device(calc_loss)

    def _place_on_good_device(self, routine, flag=False):
        if self.place_vars_on_cpu or flag:
            with tf.device('/cpu:0'):
                return routine()
        else:
            return routine()

    def _setup_loss(self):
        pass

    def _setup_train_op(self):
        LEARNING_RATE = self.learning_rate
        LEARNING_RATE_DECAY = self.learning_rate_decay
        LEARNING_RATE_DECAY_STAIRCASE = self.learning_rate_decay_staircase
        LEARNING_RATE_DECAY_STEPS = self.n_learning_rate_decay_steps

        def create_step_var():
            step_var = tf.Variable(0, trainable=False)

            self._step_var = step_var
            self._step_inc_op = step_var.assign(step_var + 1)

            return step_var

        step_var = self._place_on_good_device(create_step_var)
        self.step_var = step_var

        _params = tf.trainable_variables()
        
        if callable(self.model_exclude_params):
            _excludes = self.model_exclude_params()
            if _excludes:
                _params = list(filter(lambda x: any([item not in x.name for item in _excludes]), _params))

        with tf.name_scope('optimizer'):
            with tf.name_scope('params'):
                def create_params():
                    lr_var = tf.Variable(LEARNING_RATE, trainable=False)
                    
                    if LEARNING_RATE_DECAY and LEARNING_RATE_DECAY_STEPS:
                        lr_var = tf.train.exponential_decay(lr_var, step_var, LEARNING_RATE_DECAY_STEPS, LEARNING_RATE_DECAY, staircase=LEARNING_RATE_DECAY_STAIRCASE)

                    return lr_var

                lr_var = self._place_on_good_device(create_params)

            self._learning_rate = lr_var

            _optimizer = self._place_on_good_device(lambda: tf.train.AdamOptimizer(lr_var))
        
        with tf.name_scope('training'):
            towers_grads = self.towers_grads
            
            grads = self._place_on_good_device(lambda: sum_gradients(towers_grads))
            
#             def clip_grads(gradvars):
#                 if GRAD_CLIP_VALUE is not None:
#                     gradvars = [(tf.clip_by_norm(grad, GRAD_CLIP_VALUE), var) for grad, var in gradvars]
                    
#                 return gradvars

#             grads = self._place_on_good_device(lambda: clip_grads(grads))
            
            def get_apply_grads_op():
                apply_gradient_op = _optimizer.apply_gradients(grads, global_step=step_var)

                return apply_gradient_op

            apply_gradient_op = self._place_on_good_device(get_apply_grads_op)

            self._grads = grads
            self._params = _params
            
            self.train_op = apply_gradient_op

    def _setup_metrics(self):
        with tf.name_scope('metrics'):
            metrics = []
            if len(self._towers_outputs) > 1:
                for i, (name, outputs, batch) in enumerate(self._towers_outputs):
                    if self.place_vars_on_cpu:
                        device_setter = local_device_setter(
                            worker_device=name)
                    else:
                        device_setter = local_device_setter(
                            ps_device_type='gpu',
                            worker_device=name)#,
                            #ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                            #    len(self._gpus), tf.contrib.training.byte_size_load_fn))
                    
                    with tf.device(device_setter):
                        with tf.name_scope('tower-%i' % i):
                            result = self.metrics_getter(*batch, *outputs)
                            if result is not None:
                                if isinstance(result, (list, tuple)):
                                    metrics.append(result)
                                else:
                                    metrics.append([result])
            else:
                _, outputs, batch = self._towers_outputs[0]
                result = self.metrics_getter(*batch, *outputs)
                if result is not None:
                    if isinstance(result, (list, tuple)):
                        metrics.append(result)
                    else:
                        metrics.append([result])

            self._metrics = self._place_on_good_device(lambda: [tf.reduce_mean(m, axis=0) for m in zip(*metrics)])

    def _setup_summary(self):
        def summary_create():
            if len(self._avg_losses) > 1:
                self.train_summary_op, self.valid_summary_op = self.summary_getter(self._params, self._grads, self._learning_rate, *self._metrics, self.loss, *self._avg_losses)
            else:
                self.train_summary_op, self.valid_summary_op = self.summary_getter(self._params, self._grads, self._learning_rate, *self._metrics, self.loss)
        self._place_on_good_device(summary_create)

def sum_gradients(tower_grads):
    with tf.name_scope('gradient_summing'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_grads):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
                
        gradvars = []
        for var, grads in all_grads.items():
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.add_n(grads)
            gradvars.append((avg_grad, var))
            
        return gradvars
        