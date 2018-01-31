"""Loss functions implemented in TensorFlow."""

import tensorflow as tf

from nobrainer import scores


def dice_loss(labels, predictions):
    """Return Dice loss given two tensors. Output is in range [0, 1]."""
    return 1 - scores.dice_coefficient(labels, predictions)


def hamming_loss(labels, predictions):
    """Return Hamming loss given two tensors. Output is in range [0, 1]."""
    # QUESTION: does this implementation make sense?
    return tf.truediv(
        scores.hamming_distance(labels, predictions), tf.size(labels)
    )
