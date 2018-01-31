"""Utilities."""

import tensorflow as tf


def _shapes_equal(x1, x2):
    return tf.shape(x1) == tf.shape(x2)


def _shapes_equal_numpy(x1, x2):
    return x1.shape == x2.shape


def _check_shapes_equal(x1, x2, implementation='tf'):
    impls = {'tf': _shapes_equal, 'np': _shapes_equal_numpy}
    if not impls[implementation](x1, x2):
        raise ValueError("Shapes of both arrays or tensors must be equal.")
