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
    
    ModelBuilder = model_trainer.ModelBuilder
    Trainer = model_trainer.Trainer
    add_grads_summary = model_trainer.add_grads_summary


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
    Trainer(**trainer_options) \
        .add_inputs(lambda: (tf.placeholder(tf.float32, [None, 784], "images"),
                             tf.placeholder(tf.int32, [None], "labels"))) \
        .set_model(model) \
        .set_metrics(metrics) \
        .set_summary(summary) \
        .train(train_data_sources=[lambda inp, lbl: {inp: mnist.train.images, lbl: mnist.train.labels}],
               valid_data_sources=[lambda inp, lbl: {inp: mnist.validation.images, lbl: mnist.validation.labels}],
               verbose=True)
