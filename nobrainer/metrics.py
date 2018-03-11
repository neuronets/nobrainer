"""Metrics implemented in TensorFlow and NumPy. Functions that end in
`_numpy` are implemented in NumPy for eager computation.
"""

import numpy as np
import tensorflow as tf

from nobrainer.util import _check_all_x_in_subset_numpy, _check_shapes_equal


def dice(u, v, axis=None, name=None):
    """Return the Dice coefficients between two Tensors. Perfect Dice is 1.0.

                     2 * | intersection(A, B) |
        Dice(A, B) = --------------------------
                            |A| + |B|

    Args:
        u, v: `Tensor`, input boolean tensors.
        axis: `int` or `tuple`, the dimension(s) along which to compute Dice.
        name: `str`, a name for the operation.

    Returns:
        `Tensor` of Dice coefficient(s).

    Notes:
        This functions is similar to `scipy.spatial.distance.dice` but returns
        `1 - scipy.spatial.distance.dice(u, v)`.
    """
    with tf.name_scope(name, 'dice_coefficient', [u, v]):
        u = tf.convert_to_tensor(u)
        v = tf.convert_to_tensor(v)
        _check_shapes_equal(u, v)

        intersection = tf.reduce_sum(tf.multiply(u, v), axis=axis)
        const = tf.constant(2, dtype=intersection.dtype)
        numerator = tf.multiply(const, intersection)
        denominator = tf.add(
            tf.reduce_sum(u, axis=axis), tf.reduce_sum(v, axis=axis))
        return tf.truediv(numerator, denominator)


def dice_numpy(u, v, axis=None):
    """Return the Dice coefficients between two ndarrays. Perfect Dice is 1.0.

                     2 * | intersection(A, B) |
        Dice(A, B) = --------------------------
                            |A| + |B|

    Args:
        u, v: `ndarray`, input boolean ndarrays.
        axis: `int` or `tuple`, the dimension(s) along which to compute Dice.

    Returns:
        `float` Dice coefficient or `ndarray` of Dice coefficients.

    Notes:
        This functions is similar to `scipy.spatial.distance.dice` but returns
        `1 - scipy.spatial.distance.dice(u, v)`.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    _check_shapes_equal(u, v)

    subset = (0, 1)  # Values must be 0 or 1.
    if u.dtype != bool:
        _check_all_x_in_subset_numpy(u, subset)
    if v.dtype != bool:
        _check_all_x_in_subset_numpy(v, subset)

    numerator = 2 * (u * v).sum(axis=axis)
    denominator = u.sum(axis=axis) + v.sum(axis=axis)
    return numerator / denominator


def hamming(u, v, axis=None, name=None):
    """Return the Hamming distance between two Tensors.

    Args:
        u, v: `Tensor`, input tensors.
        axis: `int` or `tuple`, the dimension(s) along which to compute Hamming
            distance.
        name: `str`, a name for the operation.

    Returns:
        `Tensor` of Hamming distance(s).

    Notes:
        This function is almost identical to `scipy.spatial.distance.hamming`
        but accepts n-D tensors and adds an `axis` parameter.
    """
    with tf.name_scope(name, 'dice_coefficient', [u, v]):
        u = tf.convert_to_tensor(u)
        v = tf.convert_to_tensor(v)
        _check_shapes_equal(u, v)
        return tf.reduce_mean(tf.not_equal(u, v), axis=axis)


def hamming_numpy(u, v, axis=None):
    """Return Hamming distance between two ndarrays.

    Args:
        u, v: `ndarray`, input ndarrays.
        axis: `int` or `tuple`, the dimension(s) along which to compute Hamming
            distance.

    Returns:
        `float` Hamming distance or `ndarray` of Hamming distances.

    Notes:
        This function is almost identical to `scipy.spatial.distance.hamming`
        but accepts ndarrays and adds an `axis` parameter.
    """
    u_ne_v = u != v
    return np.mean(u_ne_v, axis=axis)
