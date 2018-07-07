import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

try:
    from model_trainer import Trainer, add_grads_summary
except ModuleNotFoundError:
    import importlib.util
    import os

    path = os.path.dirname(__file__)

    spec = importlib.util.spec_from_file_location("model_trainer", os.path.join(path, '..', 'src', "model_trainer.py"))
    model_trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_trainer)
    
    Trainer = model_trainer.Trainer
    add_grads_summary = model_trainer.add_grads_summary


mnist = input_data.read_data_sets("training/data/mnist/", one_hot=False)


def model(is_training_mode, images, labels, *args):
    input_layer = tf.reshape(images, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training_mode)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits, tf.argmax(input=logits, axis=1)


def loss(images, labels, logits, *args):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


def metrics(images, labels, logits, classes):
    return tf.reduce_mean(tf.cast(tf.equal(labels, tf.cast(classes, tf.int32)), tf.float32))


def summary(params, grads, learning_rate, accuracy, loss, *args):
    tf.summary.scalar('learning-rate', learning_rate)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    valid_summary_op = tf.summary.merge_all()

    add_grads_summary(grads)

    train_summary_op = tf.summary.merge_all()

    return train_summary_op, valid_summary_op

trainer_options = dict(
    n_training_steps=100000,
    training_dir_path='./training/mnist'
)

Trainer(**trainer_options) \
    .set_inputs(lambda: (tf.placeholder(tf.float32, [None, 784], "images"),
                         tf.placeholder(tf.int32, [None], "labels"))) \
    .set_model(model) \
    .set_loss(loss) \
    .set_metrics(metrics) \
    .set_summary(summary) \
    .train(train_data_source=lambda inp, lbl: {inp: mnist.train.images, lbl: mnist.train.labels},
           valid_data_source=lambda inp, lbl: {inp: mnist.validation.images, lbl: mnist.validation.labels},
           verbose=True)
