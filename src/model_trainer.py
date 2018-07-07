import os
import shutil
import tensorflow as tf
import time
from tensorflow.python.client import device_lib


def add_grads_summary(grads):
    for grad, var in grads:
        if grad is not None:
            tf.summary.scalar('gradients/' + var.op.name, tf.norm(grad))
            tf.summary.histogram('gradients/' + var.op.name, grad)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
                       vars_moving_average_decay=0.9999):
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

        self.vars_moving_average_decay = vars_moving_average_decay

        self._is_builded = False
        self.saver = None

    def set_inputs(self, dataset_placeholders_getter, dataset_mapper=None, flat_mapper=False):
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

        self.dataset_placeholders_getter = dataset_placeholders_getter
        # self.dataset_output_dtypes = output_dtypes
        # self.dataset_output_shapes = output_shapes
        self.dataset_needs_flatting = flat_mapper
        self.dataset_mapper = dataset_mapper
        
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
    
    def train(self, train_data_source, valid_data_source, model_initial_weights_loader=None, pre_start_hooks=[], pre_train_hooks=[], post_train_hooks=[], pre_end_hooks=[], verbose=False, training_dir_path=None):
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

        gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            try:
                if verbose:
                    print('Initializing parameters ', flush=True, end='')
                
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                train_iter_feed_dict = dict()
                train_iter_feed_dict.update(train_data_source(*self.dataset_placeholders))
                if self.pipe_name_tf_phr is not None:
                    train_iter_feed_dict[self.pipe_name_tf_phr] = 'train'

                sess.run(self.train_iterator.initializer, train_iter_feed_dict)

                valid_iter_feed_dict = dict()
                valid_iter_feed_dict.update(valid_data_source(*self.dataset_placeholders))
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

                    if verbose:
                        print('Initial train loss = %.6f, valid loss = %.6f.' % (_train_loss, _valid_loss), flush=True)

                if verbose:
                    print('Start trainging.', flush=True)
                
                if verbose:
                    start = time.time()
                for _ in range(_step, TRAINING_STEPS):
                    _step = int(sess.run(self._step_inc_op))
                    # sess.run(self._step_var))

                    if pre_train_hooks:
                        for item in pre_train_hooks:
                            if callable(item):
                                item(sess, _step)

                    sess.run([self.train_op], {self.is_training_mode: True, self.data_loader_mode: 'train-pipe'})

                    if post_train_hooks:
                        for item in post_train_hooks:
                            if callable(item):
                                item(sess, _step)
                    
                    if _step % STEPS_PER_SUMMARY == 0:
                        _train_loss, _train_summary = sess.run([self.loss, self.train_summary_op], {self.data_loader_mode: 'train-pipe'})
                        _valid_loss, _valid_summary = sess.run([self.loss, self.valid_summary_op], {self.data_loader_mode: 'valid-pipe'})
                        _train_summary_writer.add_summary(_train_summary, _step)
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
        NEEDS_FLATTING = self.dataset_needs_flatting

        dataset_placeholders = self.dataset_placeholders_getter()
        if not isinstance(dataset_placeholders, (tuple, list)):
            raise ValueError('dataset_placeholders: is neither a tuple nor a list')

        if ENABLE_CACHING and CACHE_DIR_PATH is not None:
            pipe_name_tf_phr = tf.placeholder(tf.string, name='pipe_name')
        else:
            pipe_name_tf_phr = None

        dataset = tf.data.Dataset().from_tensor_slices(dataset_placeholders)

        if self.dataset_mapper is not None:
            dataset = dataset.map(self.dataset_mapper, DATASET_N_WORKERS)
            dataset = dataset.apply(tf.contrib.data.ignore_errors())
        
        if ENABLE_CACHING:
            if CACHE_DIR_PATH is not None:
                if not os.path.exists(CACHE_DIR_PATH):
                    os.makedirs(CACHE_DIR_PATH)
                dataset = dataset.cache(tf.constant(CACHE_DIR_PATH) + pipe_name_tf_phr)
            else:
                dataset = dataset.cache()
            
        if NEEDS_FLATTING:
            dataset = dataset.flat_map(lambda *samples: tf.data.Dataset.from_tensor_slices(samples))

        # dataset = dataset.map(lambda *sample: tuple(tf.reshape(item, shape) for item, shape in zip(sample, self.dataset_output_shapes)), DATASET_N_WORKERS)
        dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=BATCH_SIZE*N_TRAINERS)
        dataset = dataset.prefetch(buffer_size=1)

        self.pipe_name_tf_phr = pipe_name_tf_phr

        self.dataset_placeholders = dataset_placeholders
        self.dataset = dataset

        self.train_iterator = dataset.make_initializable_iterator('train')
        self.valid_iterator = dataset.make_initializable_iterator('valid')

        self.train_batch = self.train_iterator.get_next()
        self.valid_batch = self.valid_iterator.get_next()

    def _setup_model(self):
        with tf.name_scope('aux'):
            is_training_mode = tf.placeholder_with_default(False, [], name='is_training_mode')
            data_loader_mode = tf.placeholder_with_default('train-pipe', [], name='data_loader_mode')
            
            _batch = tf.case([(tf.equal(data_loader_mode, 'train-pipe'), lambda: self.train_batch),
                              (tf.equal(data_loader_mode, 'valid-pipe'), lambda: self.valid_batch)],
                             exclusive=True)
            _batch = tuple(tf.reshape(_batch[i], [-1] + shape[1:].as_list()) for i, shape in enumerate(self.dataset.output_shapes))
        
        self._towers_outputs = []
        if len(self._gpus) > 1:
            with tf.variable_scope(tf.get_variable_scope()):
                _batch = list(zip(*tuple(tf.split(_batch[i], [self.batch_size]*len(self._gpus)) for i in range(len(_batch)))))
                for i, name in enumerate(self._gpus):
                    with tf.device(name):
                        with tf.name_scope('tower-%i' % i):
                            outputs = self.model_getter(is_training_mode, *_batch[i])
                            self._towers_outputs.append((name, outputs, _batch[i]))
                            tf.get_variable_scope().reuse_variables()
        else:
            outputs = self.model_getter(is_training_mode, *_batch)
            self._towers_outputs.append((None, outputs, _batch))

        self.is_training_mode = is_training_mode
        self.data_loader_mode = data_loader_mode
            

    def _setup_loss(self):
        self._towers_losses = []
        avg_losses = []
        with tf.name_scope('losses'):
            if len(self._towers_outputs) > 1:
                for i, (name, outputs, batch) in enumerate(self._towers_outputs):
                    with tf.device(name):
                        with tf.name_scope('tower-%i' % i) as scope:
                            _ = self.loss_getter(*batch, *outputs)
                            losses = tf.losses.get_losses(scope)
                            reg_losses = tf.losses.get_regularization_losses(scope)
                            self._towers_losses.append((name, losses, reg_losses))
                            avg_losses.append(losses)
            else:
                _, outputs, batch = self._towers_outputs[0]
                _ = self.loss_getter(*batch, *outputs)
                losses = tf.losses.get_losses()
                reg_losses = tf.losses.get_regularization_losses()
                self._towers_losses.append((None, losses, reg_losses))
                avg_losses.append(losses)

            self._avg_losses = [tf.reduce_mean(l, axis=0) for l in zip(*avg_losses)]
            self.loss = tf.add_n(self._avg_losses)

    def _setup_train_op(self):
        LEARNING_RATE = self.learning_rate
        LEARNING_RATE_DECAY = self.learning_rate_decay
        LEARNING_RATE_DECAY_STAIRCASE = self.learning_rate_decay_staircase
        LEARNING_RATE_DECAY_STEPS = self.n_learning_rate_decay_steps
        GRAD_CLIP_VALUE = self.grad_clip_value
        MOVING_AVERAGE_DECAY = self.vars_moving_average_decay

        _params = tf.trainable_variables()

        step_var = tf.Variable(0, trainable=False)

        self._step_var = step_var
        self._step_inc_op = step_var.assign(step_var + 1)
        
        if callable(self.model_exclude_params):
            _excludes = self.model_exclude_params()
            if _excludes:
                _params = list(filter(lambda x: any([item not in x.name for item in _excludes]), _params))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope('optimizer'):
            with tf.name_scope('params'):
                lr_var = tf.Variable(LEARNING_RATE, trainable=False)
                
                if LEARNING_RATE_DECAY and LEARNING_RATE_DECAY_STEPS:
                    lr_var = tf.train.exponential_decay(lr_var, step_var, LEARNING_RATE_DECAY_STEPS, LEARNING_RATE_DECAY, staircase=LEARNING_RATE_DECAY_STAIRCASE)

            self._learning_rate = lr_var

            _optimizer = tf.train.AdamOptimizer(lr_var)

            towers_grads = []
            if len(self._towers_losses) > 1:
                for i, (name, losses, reg_losses) in enumerate(self._towers_losses):
                    with tf.device(name):
                        with tf.name_scope('tower-%i' % i):
                            assert len(losses), 'Losses aren\'t provided'

                            loss = tf.add_n(losses)
                            if len(reg_losses):
                                loss = loss + tf.add_n(reg_losses)

                            grads = _optimizer.compute_gradients(loss)
                            towers_grads.append(grads)
            else:
                _, losses, reg_losses = self._towers_losses[0]
                assert len(losses), 'Losses aren\'t provided'

                loss = tf.add_n(losses)
                if len(reg_losses):
                    loss = loss + tf.add_n(reg_losses)

                grads = _optimizer.compute_gradients(loss)
                towers_grads.append(grads)

            grads = []
            for grad_and_vars in zip(*towers_grads):
                _grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)

                    _grads.append(expanded_g)

                grad = tf.concat(axis=0, values=_grads)
                grad = tf.reduce_mean(grad, 0)

                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                grads.append(grad_and_var)

            if GRAD_CLIP_VALUE is not None:
                grads = [(tf.clip_by_norm(grad, GRAD_CLIP_VALUE), var) for grad, var in grads]

            apply_gradient_op = _optimizer.apply_gradients(grads)
            
            params_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, step_var)
            params_averages_op = params_averages.apply(_params)

            self._grads = grads
            self._params_averages = params_averages
            self.train_op = tf.group(apply_gradient_op, params_averages_op, update_ops)

    def _setup_metrics(self):
        with tf.name_scope('metrics'):
            metrics = []
            if len(self._towers_outputs) > 1:
                for i, (name, outputs, batch) in enumerate(self._towers_outputs):
                    with tf.device(name):
                        with tf.name_scope('tower-%i' % i):
                            result = self.metrics_getter(*batch, *outputs)
                            if isinstance(result, (list, tuple)):
                                metrics.append(result)
                            else:
                                metrics.append([result])
            else:
                _, outputs, batch = self._towers_outputs[0]
                result = self.metrics_getter(*batch, *outputs)
                if isinstance(result, (list, tuple)):
                    metrics.append(result)
                else:
                    metrics.append([result])

            self._metrics = [tf.reduce_mean(m, axis=0) for m in zip(*metrics)]

    def _setup_summary(self):
        self.train_summary_op, self.valid_summary_op = self.summary_getter(self._params_averages, self._grads, self._learning_rate, *self._metrics, *self._avg_losses)
