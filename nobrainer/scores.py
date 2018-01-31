"""Scoring functions implemented in TensorFlow and NumPy. Functions that end in
`_numpy` are implemented in NumPy for convenience.
"""

import numpy as np
import tensorflow as tf

from nobrainer import utils


def dice_coefficient(labels, predictions):
    """Return Dice coefficient between two tensors.

                 2 * |A * B|
    Dice(A, B) = -----------
                  |A| + |B|
    """
    utils._check_shapes_equal(labels, predictions)
    labels = tf.reshape(labels, -1)
    predictions = tf.reshape(predictions, -1)
    return (
        2 * tf.reduce_sum(tf.multiply(labels, predictions))
        / (tf.reduce_sum(labels) + tf.reduce_sum(predictions))
    )


def dice_coefficient_numpy(labels, predictions):
    """Return Dice coefficient between two Numpy arrays.

                2 * |A * B|
    Dice(A,B) = -----------
                 |A| + |B|
    """
    utils._check_shapes_equal(labels, predictions, implementation='np')
    labels = labels.flatten()
    predictions = predictions.flatten()
    return (
        2 * (labels * predictions).sum() / (labels.sum() + predictions.sum())
    )


def hamming_distance(labels, predictions):
    """Return Hamming distance between two tensors."""
    utils._check_shapes_equal(labels, predictions)
    return tf.reduce_sum(tf.not_equal(labels, predictions))


def hamming_distance_numpy(labels, predictions):
    """Return Hamming distance between two tensors."""
    utils._check_shapes_equal(labels, predictions, implementation='np')
    return np.not_equal(labels, predictions).sum()
