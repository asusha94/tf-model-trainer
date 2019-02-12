
import tensorflow as tf


def add_grads_summary(grads, scope=None):
    with tf.name_scope(scope, 'gradients'):
        for grad, var in grads:
            if grad is not None:
                grad_ = tf.boolean_mask(grad, tf.is_finite(grad))
                tf.summary.scalar(var.op.name, tf.norm(grad_))
                tf.summary.histogram(var.op.name, grad_)
