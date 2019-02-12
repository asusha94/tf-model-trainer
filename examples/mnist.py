from tf_trainer.summary import add_grads_summary
from tf_trainer import ModelBuilder, Trainer, Dataset, DatasetIteratorNames
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys

path = os.path.dirname(__file__)

sys.path = [os.path.join(path, '..', 'src')] + sys.path


mnist = input_data.read_data_sets("training/data/mnist/", one_hot=False)


def forward(is_training_mode, images, labels):
    input_layer = tf.reshape(images, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv1')

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='dense1')
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training_mode)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10, name='dense2')

    return logits


def loss(images, labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


def metrics(model):
    _, labels = model.inputs
    logits = model.outputs
    classes = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(labels, tf.cast(classes, tf.int32)), tf.float32))


def summary(model, step, learning_rate, grads, metrics):
    tf.summary.scalar('accuracy', metrics)
    tf.summary.scalar('learning-rate', learning_rate)
    tf.summary.scalar('loss', tf.add_n(model.losses))

    valid_summary_op = tf.summary.merge_all()

    add_grads_summary(grads)

    train_summary_op = tf.summary.merge_all()

    return train_summary_op, valid_summary_op


def feed_dict(state, inputs, labels):
    if state == DatasetIteratorNames.Training:
        return {inputs: mnist.train.images, labels: mnist.train.labels}
    elif state == DatasetIteratorNames.Validation:
        return {inputs: mnist.validation.images, labels: mnist.validation.labels}


model = ModelBuilder() \
    .set_forward(forward) \
    .set_loss(loss) \
    .build()

trainer_options = dict(
    n_training_steps=100000,
    training_dir_path='./training/mnist',
    place_vars_on_cpu=False,
    batch_size=128
)

if __name__ == '__main__':
    dataset = Dataset(batch_size=8).add_source(lambda: (tf.placeholder(tf.float32, [None, 784], "images"),
                                                        tf.placeholder(tf.int32, [None], "labels")),
                                               feed_dict)
    dataset.compile()

    train_batch = dataset.outputs(DatasetIteratorNames.Training).get_next()
    valid_batch = dataset.outputs(DatasetIteratorNames.Validation).get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dataset.init(sess)

        b = sess.run(train_batch)
        print('train', len(b), b[0].shape)

        b = sess.run(valid_batch)
        print('valid', len(b), b[0].shape)

    Trainer(**trainer_options) \
        .add_dataset(lambda: (tf.placeholder(tf.float32, [None, 784], "images"),
                              tf.placeholder(tf.int32, [None], "labels")),
                     feed_dict) \
        .set_model(model) \
        .set_metrics(metrics) \
        .set_summary(summary) \
        .train(verbose=True)
