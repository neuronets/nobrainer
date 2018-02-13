"""Scoring functions implemented in TensorFlow and NumPy. Functions that end in
`_numpy` are implemented in NumPy for eager computation.
"""

import numpy as np
import tensorflow as tf

from nobrainer.util import _check_shapes_equal


def dice_coefficient(x1, x2, reducer=tf.reduce_mean):
    """Return the Dice coefficients between two Tensors. Assumes that the first
    axis is the batch.

                     2 * | intersection(A, B) |
        Dice(A, B) = --------------------------
                            |A| + |B|

    Parameters
    ----------
    x1, x2 : Tensor
        Tensors of data. The first axis is assumed to be the batch.
    reducer : callable
        If this is None, Dice coefficient is calculated for each item in the
        batch. If this is a callable, the array of Dice coefficients is reduced
        using this callable.

    Returns
    -------
    Tensor of Dice coefficients if `reducer` is None, otherwise single Dice
    coefficient.
    """
    with tf.name_scope('dice_coefficient'):
        x1 = tf.contrib.layers.flatten(x1)
        x2 = tf.contrib.layers.flatten(x2)
        _check_shapes_equal(x1, x2)

        dice_by_elem = (
            2 * tf.reduce_sum(x1 * x2, -1)
            / (tf.reduce_sum(x1, -1) + tf.reduce_sum(x2, -1))
        )

        return reducer(dice_by_elem) if reducer is not None else dice_by_elem


def dice_coefficient_numpy(x1, x2, reducer=np.mean):
    """Return the Dice coefficients between two Numpy arrays. Assumes that the
    first axis is the batch.

                     2 * | intersection(A, B) |
        Dice(A, B) = --------------------------
                            |A| + |B|

    Parameters
    ----------
    x1, x2 : ndarray
        Arrays of data. The first axis is assumed to be the batch.
    reducer : callable
        If this is None, Dice coefficient is calculated for each item in the
        batch. If this is a callable, the array of Dice coefficients is reduced
        using this callable.

    Returns
    -------
    Array of Dice coefficients if `reducer` is None, otherwise single Dice
    coefficient.
    """
    batch_size = x1.shape[0]
    x1 = x1.reshape(batch_size, -1)
    x2 = x2.reshape(batch_size, -1)
    _check_shapes_equal(x1, x2)

    dice_by_elem = 2 * (x1 * x2).sum(-1) / (x1.sum(-1) + x2.sum(-1))

    return reducer(dice_by_elem) if reducer is not None else dice_by_elem


def dice_coefficient_by_class_numpy(x1, x2, reducer=np.mean):
    """Return Dice coefficient per class. Class labels must be in last
    dimension.

    Parameters
    ----------
    x1, x2 : ndarray
        Arrays of data. The first axis is assumed to be the batch.
    reducer : callable
        If this is None, Dice coefficient is calculated for each item in the
        batch, per class. If this is a callable, the array of Dice coefficients
        per class is reduced using this callable.

    Returns
    -------
    Array of Dice coefficients, one per class. If
    """
    _check_shapes_equal(x1, x2)
    num_classes = x1.shape[-1]

    _shape = (num_classes, x1.shape[0]) if reducer is None else num_classes
    coefficients = np.zeros(_shape, np.float32)

    for ii in range(num_classes):
        coefficients[ii] = dice_coefficient_numpy(
            x1=x1[Ellipsis, ii],
            x2=x2[Ellipsis, ii],
            reducer=reducer,
        )

    return coefficients


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
