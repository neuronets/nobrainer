"""Scoring functions implemented in TensorFlow and NumPy. Functions that end in
`_numpy` are implemented in NumPy for convenience.
"""

import numpy as np
import tensorflow as tf

from nobrainer.utils import _check_shapes_equal


def dice_coefficient(labels, predictions):
    """Return Dice coefficient between two tensors.

                 2 * |A * B|
    Dice(A, B) = -----------
                  |A| + |B|
    """
    with tf.name_scope('dice_coefficient'):
        labels = tf.contrib.layers.flatten(labels)
        predictions = tf.contrib.layers.flatten(predictions)
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
    _check_shapes_equal(labels, predictions)
    labels = labels.flatten()
    predictions = predictions.flatten()
    return (
        2 * (labels * predictions).sum() / (labels.sum() + predictions.sum())
    )


def dice_coefficient_by_class_numpy(labels, predictions, num_classes):
    """Return Dice coefficient per class. Class labels must be in last
    dimension.
    """
    coefficients = np.zeros(num_classes, np.float32)

    # QUESTION: can this be done in vectorized fashion?
    for ii in range(num_classes):
        coefficients[ii] = dice_coefficient_numpy(
            labels=labels[Ellipsis, ii],
            predictions=predictions[Ellipsis, ii])

    return coefficients.astype(np.float32)


def hamming_distance(labels, predictions):
    """Return Hamming distance between two tensors."""
    with tf.name_scope('hamming_distance'):
        _check_shapes_equal(labels, predictions)
        return tf.reduce_sum(
            tf.cast(tf.not_equal(labels, predictions), tf.int32)
        )


def hamming_distance_numpy(labels, predictions):
    """Return Hamming distance between two Numpy arrays."""
    _check_shapes_equal(labels, predictions)
    return np.not_equal(labels, predictions).sum()
