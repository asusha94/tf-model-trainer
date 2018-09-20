import tensorflow as tf
from tensorflow.keras.datasets import cifar10

import os
import sys

path = os.path.dirname(__file__)

sys.path = [os.path.join(path, '..', 'src')] + sys.path

from tf_trainer import ModelBuilder, Trainer
from tf_trainer.summary import add_grads_summary


def _residual_v1(x,
                 kernel_size,
                 in_filter,
                 out_filter,
                 stride,
                 training,
                 activate_before_residual=False):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    with tf.name_scope('residual_v1') as name_scope:
        orig_x = x
        
        x = tf.layers.conv2d(
            inputs=x,
            filters=out_filter,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            activation=None)

        x = tf.layers.batch_normalization(x, training=training)

        x = tf.nn.relu(x)
        
        x = tf.layers.conv2d(
            inputs=x,
            filters=out_filter,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation=None)

        x = tf.layers.batch_normalization(x, training=training)

        if in_filter != out_filter:
            with tf.name_scope('avg_pool') as name_scope:
                orig_x = tf.layers.average_pooling2d(
                            orig_x, stride, stride, 'SAME')
            pad = (out_filter - in_filter) // 2
            orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

        x = tf.nn.relu(tf.add(x, orig_x))

        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x
    
def forward(is_training_mode, images, labels):
    filters = [16, 16, 32, 64]
    strides = [1, 2, 2]
    num_layers = 44
    
    input_layer = tf.cast(tf.reshape(images, [-1, 32, 32, 3]), tf.float32)
    
    input_layer = input_layer / 128 - 1

    # Convolutional Layer #1
    x = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=None,
        use_bias=False)
    
    x = tf.layers.batch_normalization(x, training=is_training_mode)
    
    x = tf.nn.relu(x)
    
    for i in range(3):
        with tf.name_scope('stage'):
            for j in range((num_layers - 2) // 6):
                if j == 0:
                    # First block in a stage, filters and strides may change.
                    x = _residual_v1(x, 3, filters[i], filters[i + 1], strides[i], training=is_training_mode)
                else:
                    # Following blocks in a stage, constant filters and unit stride.
                    x = _residual_v1(x, 3, filters[i + 1], filters[i + 1], 1, training=is_training_mode)

    with tf.name_scope('global_avg_pool'):
        assert x.get_shape().ndims == 4
        x = tf.reduce_mean(x, [1, 2])
        
    with tf.name_scope('fully_connected'):
        x = tf.layers.dense(x, 10)

    return x


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


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def feed_dict(state, inputs, labels):
    if state == 'train':
        return { inputs: x_train, labels: y_train }
    elif state == 'valid':
        return { inputs: x_test, labels: y_test }


tf.set_random_seed(42)


model = ModelBuilder() \
    .set_forward(forward) \
    .set_loss(loss) \
    .build()

trainer_options = dict(
    n_training_steps=100000,
    training_dir_path='./training/cifar10',
    place_vars_on_cpu=False,
    batch_size=96,
    buffer_size=96*4
)

if __name__ == '__main__':
    Trainer(**trainer_options) \
        .add_dataset(lambda: (tf.placeholder(tf.float32, [None, 32, 32, 3], "images"),
                              tf.placeholder(tf.int32, [None, 1], "labels")),
                     feed_dict) \
        .set_model(model) \
        .set_metrics(metrics) \
        .set_summary(summary) \
        .train(verbose=True,
               auto_freeze=dict(input_getter=lambda: (tf.placeholder(tf.float32, [None, 32, 32, 3], "images"), None)))
