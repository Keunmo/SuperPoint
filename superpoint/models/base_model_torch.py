from abc import ABCMeta, abstractmethod
import tensorflow as tf
import torch
import numpy as np
from tqdm import tqdm

# TODO: implement multi-gpus training (cf. cifar loss tower)


class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


class BaseModel(metaclass=ABCMeta):
    dataset_names = set(['training', 'validation', 'test'])
    required_baseconfig = ['batch_size', 'learning_rate']

    def __init__(self, data={}, use_gpu=True, data_shape=None, **config):
        self.datasets = data
        self.data_shape = data_shape
        self.use_gpu = use_gpu
        # self.graph = tf.get_default_graph()
        # self.name = self.__class__.__name__.lower()  # get child name

        # Update config
        self.config = getattr(self, 'default_config', {})
        self.config.update(config)

        required = self.required_baseconfig + getattr(self, 'required_config_keys', [])
        for r in required:
            assert r in self.config, 'Required configuration entry: \'{}\''.format(r)
        assert set(self.datasets) <= self.dataset_names, \
            'Unknown dataset name: {}'.format(set(self.datasets)-self.dataset_names)
        assert self.datasets or (data_shape is not None)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._build_graph()

    def _train_graph(self, data):
        with tf.name_scope('train'):
            net_outputs = self._model(data, Mode.TRAIN, training=True, **self.config)
            self.loss = self._loss(net_outputs, data, **self.config)
            tf.summary.scalar('loss', self.loss)

    def _eval_graph(self, data):
        with tf.name_scope('eval'):
            net_outputs = self._model(data, Mode.EVAL, training=False, **self.config)
            self.metrics = self._metrics(net_outputs, data, **self.config)

    def _pred_graph(self, data):
        with tf.name_scope('pred'):
            self.pred_out = self._model(data, Mode.PRED, training=False, **self.config)

    def _build_graph(self):
        # Training and evaluation network, if tf datasets provided
        if self.datasets:
            # Generate iterators for the given tf datasets
            self.dataset_iterators = {}
            for n, d in self.datasets.items():
                if n == 'training':
                    d = d.repeat().batch(self.config['batch_size'])
                    self.dataset_iterators[n] = d.make_one_shot_iterator()
                else:
                    d = d.batch(self.config.get('eval_batch_size', 1))
                    self.dataset_iterators[n] = d.make_initializable_iterator()
                output_types = d.output_types
                output_shapes = d.output_shapes
                self.datasets[n] = d

                # Perform compatibility checks with the inputs of the child model
                for i, spec in self.input_spec.items():
                    assert i in output_shapes
                    tf.TensorShape(output_shapes[i]).assert_is_compatible_with(
                            tf.TensorShape(spec['shape']))

            # Used for input shapes of the prediction network
            if self.data_shape is None:
                self.data_shape = output_shapes

            # Handle for the feedable iterator
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                    self.handle, output_types, output_shapes)
            data = iterator.get_next()

            # Build the actual training and evaluation models
            self._train_graph(data)
            self._eval_graph(data)
            self.summaries = tf.summary.merge_all()

            # Create training ops
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            opt = tf.train.RMSPropOptimizer(self.config['learning_rate'])
            with tf.control_dependencies(update_ops):
                self.trainer = opt.minimize(self.loss, global_step=self.global_step)

        # Prediction network with feed_dict
        self.pred_in = {i: tf.placeholder(spec['type'], shape=self.data_shape[i])
                        for i, spec in self.input_spec.items()}
        self._pred_graph(self.pred_in)

        # Start session
        sess_config = tf.ConfigProto(
                device_count={'GPU': 0} if not self.use_gpu else {})  # TODO: multi-gpus
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        # Register tf dataset handles
        if self.datasets:
            self.dataset_handles = {}
            for n, i in self.dataset_iterators.items():
                self.dataset_handles[n] = self.sess.run(i.string_handle())

        self.sess.run([tf.global_variables_initializer(),
                       tf.local_variables_initializer()])
        self.saver = tf.train.Saver(save_relative_paths=True)
        self.graph.finalize()

    def train(self, iterations, validation_interval=100, output_dir=None):
        assert 'training' in self.datasets, 'Training dataset is required.'
        if output_dir is not None:
            train_writer = tf.summary.FileWriter(output_dir)

        tf.logging.info('Start training')
        for i in range(iterations):
            loss, summaries, _ = self.sess.run(
                    [self.loss, self.summaries, self.trainer],
                    feed_dict={self.handle: self.dataset_handles['training']})

            if 'validation' in self.datasets and i % validation_interval == 0:
                metrics = self.evaluate('validation', mute=True)
                tf.logging.info("Iter {:4d}: loss {:.4f}, accuracy {:.4f}".format(
                    i, loss, metrics['accuracy']))

                if output_dir is not None:
                    train_writer.add_summary(summaries, i)
                    metrics_summaries = tf.Summary(value=[
                        tf.Summary.Value(tag=m, simple_value=v)
                        for m, v in metrics.items()])
                    train_writer.add_summary(metrics_summaries, i)
        tf.logging.info('Training finished')

    def predict(self, data, keys='pred', batch=False):
        assert set(data.keys()) >= set(self.input_spec.keys())
        if isinstance(keys, str):
            if keys == '*':
                op = self.pred_out  # just gather all outputs
            else:
                op = self.pred_out[keys]
        else:
            op = {k: self.pred_out[k] for k in keys}
        if not batch:  # add batch dimension
            data = {d: [v] for d, v in data.items()}
        feed = {self.pred_in[i]: data[i] for i in self.input_spec}
        pred = self.sess.run(op, feed_dict=feed)
        return pred if batch else pred[0]

    def evaluate(self, dataset, max_iterations=None, mute=False):
        assert dataset in self.datasets
        self.sess.run(self.dataset_iterators[dataset].initializer)

        if not mute:
            tf.logging.info('Starting evaluation of dataset \'{}\''.format(dataset))
            if max_iterations:
                pbar = tqdm(total=max_iterations, ascii=True)
        i = 0
        metrics = []
        while True:
            try:
                metrics.append(self.sess.run(self.metrics,
                               feed_dict={self.handle: self.dataset_handles[dataset]}))
            except tf.errors.OutOfRangeError:
                break
            if max_iterations:
                i += 1
                if not mute:
                    pbar.update(1)
                if i == max_iterations:
                    break
        if not mute:
            tf.logging.info('Finished evaluation')
            if max_iterations:
                pbar.close()

        # List of dicts to dict of lists
        metrics = dict(zip(metrics[0], zip(*[m.values() for m in metrics])))
        metrics = {m: np.nanmean(metrics[m], axis=0) for m in metrics}
        return metrics

    def load(self, checkpoint_path, last=True):
        if last:
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        self.saver.restore(self.sess, checkpoint_path)

    def save(self, checkpoint_path):
        step = self.sess.run(self.global_step)
        tf.logging.info('Saving checkpoint for iteration #{}'.format(step))
        self.saver.save(self.sess, checkpoint_path, write_meta_graph=False,
                        global_step=step)

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @abstractmethod
    def _model(self, inputs, **config):
        raise NotImplementedError

    @abstractmethod
    def _loss(self, outputs, inputs, **config):
        raise NotImplementedError

    @abstractmethod
    def _metrics(self, outputs, inputs, **config):
        raise NotImplementedError
