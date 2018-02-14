"""Scoring functions implemented in TensorFlow and NumPy. Functions that end in
`_numpy` are implemented in NumPy for eager computation.
"""

import numpy as np
import tensorflow as tf

from nobrainer.util import _check_all_x_in_subset_numpy, _check_shapes_equal


def dice_coefficient(x1, x2, reducer=tf.reduce_mean, name=None):
    """Return the Dice coefficients between two Tensors. Assumes that the first
    axis is the batch.

                     2 * | intersection(A, B) |
        Dice(A, B) = --------------------------
                            |A| + |B|

    Args:
        x1, x2 : `Tensor`, input tensors with values 0 or 1. The first axis is
            assumed to be the batch.
        reducer : `callable`, if this is None, Dice coefficient is calculated
            for each item in the batch. If this is a callable, the array of
            Dice coefficients is reduced using this callable.

    Returns:
        `Tensor` of Dice coefficients of shape `(batch_size,)` if `reducer` is
        None, otherwise `Tensor` of reduced Dice coefficient.
    """
    with tf.name_scope(name, 'dice_coefficient', [x1, x2]):
        x1 = tf.convert_to_tensor(x1)
        x2 = tf.convert_to_tensor(x2)
        x1 = tf.contrib.layers.flatten(x1)  # preserves batch dimension
        x2 = tf.contrib.layers.flatten(x2)
        _check_shapes_equal(x1, x2)

        dice_by_elem = (
            2 * tf.reduce_sum(x1 * x2, -1)
            / (tf.reduce_sum(x1, -1) + tf.reduce_sum(x2, -1))
        )

        # TODO:
        #   - Account for nan output (zero denominator)
        #   - Check that all values are either 0 or 1.

        return reducer(dice_by_elem) if reducer is not None else dice_by_elem


def dice_coefficient_numpy(x1, x2, reducer=np.mean):
    """Return the Dice coefficients between two Numpy arrays. Assumes that the
    first axis is the batch.

                     2 * | intersection(A, B) |
        Dice(A, B) = --------------------------
                            |A| + |B|

    Args:
        x1, x2 : `ndarray`, arrays of data with values 0 or 1. The first axis
            is assumed to be the batch.
        reducer : `callable`, if this is None, Dice coefficient is calculated
            for each item in the batch. If this is a callable, the array of
            Dice coefficients is reduced using this callable.

    Returns:
        `ndarray` of Dice coefficients of shape `(batch_size,)` if `reducer` is
        None, otherwise reduced Dice coefficient.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    batch_size = x1.shape[0]
    x1 = x1.reshape(batch_size, -1)
    x2 = x2.reshape(batch_size, -1)
    _check_shapes_equal(x1, x2)

    subset = (0, 1)  # Values must be 0 or 1.
    _check_all_x_in_subset_numpy(x1, subset)
    _check_all_x_in_subset_numpy(x2, subset)

    # QUESTION: what should the behavior be if the denominator is 0?
    dice_by_elem = 2 * (x1 * x2).sum(-1) / (x1.sum(-1) + x2.sum(-1))

    np.nan_to_num(dice_by_elem, copy=False)

    return reducer(dice_by_elem) if reducer is not None else dice_by_elem


def dice_coefficient_by_class_numpy(x1, x2, reducer=np.mean):
    """Return Dice coefficient per class. Assumes that the first axis is the
    batch and that class labels are in the last axis.

    Args:
        x1, x2 : `ndarray`, arrays of data with values 0 or 1. The first axis
            is assumed to be the batch, and the last axis is assumed to contain
            one-hot encoded classes. This means that the size of the last axis
            must equal the number of classes.
        reducer : `callable`, if None, Dice coefficient is calculated for each
            item in the batch, per class. If this is a callable, the array of
            Dice coefficients within each class is reduced using this callable.

    Returns: `ndarray` of Dice coefficients, one per class. If `reducer` is
        None, the shape of the output is `(n_classes, batch_size)`. Otherwise,
        the shape of the output is `(n_classes,)`.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
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


def hamming_distance(x1, x2, reducer=tf.reduce_mean, name=None):
    """Return Hamming distance between two tensors.

    Args:
        x1, x2: `Tensor`, input tensors. The first axis is assumed to be the
            batch.
        reducer : `callable`, if this is None, Hamming distance is calculated
            for each item in the batch. If this is a callable, the tensor of
            Hamming distances is reduced using this callable.

    Returns:
        `Tensor` of Hamming distances of shape `(batch_size,)` if `reducer` is
        None, otherwise `Tensor` of reduced Hamming distance.
    """
    with tf.name_scope(name, 'hamming_distance', [x1, x2]):
        x1 = tf.convert_to_tensor(x1)
        x2 = tf.convert_to_tensor(x2)
        _check_shapes_equal(x1, x2)
        hamming_by_elem = tf.count_nonzero(tf.not_equal(x1, x2), axis=-1)
        return (
            reducer(hamming_by_elem) if reducer is not None
            else hamming_by_elem
        )


def hamming_distance_numpy(x1, x2, reducer=np.mean):
    """Return Hamming distance between two tensors.

    Args:
        x1, x2: `ndarray`, input arrays. The first axis is assumed to be the
            batch.
        reducer : `callable`, if this is None, Hamming distance is calculated
            for each item in the batch. If this is a callable, the array of
            Hamming distances is reduced using this callable.

    Returns:
        `ndarray` of Hamming distances of shape `(batch_size,)` if `reducer` is
        None, otherwise value of reduced Hamming distance.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    batch_size = x1.shape[0]
    x1 = x1.reshape(batch_size, -1)
    x2 = x2.reshape(batch_size, -1)
    _check_shapes_equal(x1, x2)

    hamming_by_elem = np.not_equal(x1, x2).sum(-1)

    return reducer(hamming_by_elem) if reducer is not None else hamming_by_elem
