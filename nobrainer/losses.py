"""Loss functions implemented in TensorFlow."""

import tensorflow as tf

from nobrainer import metrics


def dice_loss(labels, predictions):
    """Return Dice loss given two tensors. Output is in range [0, 1]."""
    with tf.name_scope('dice_loss'):
        best = tf.constant(1., dtype=tf.float32)
        return tf.subtract(best, metrics.dice_coefficient(labels, predictions))


def hamming_loss(labels, predictions):
    """Return Hamming loss given two tensors. Output is in range [0, 1]."""
    # QUESTION: does this implementation make sense?
    with tf.name_scope('hamming_loss'):
        return tf.truediv(
            metrics.hamming_distance(labels, predictions), tf.size(labels)
        )
