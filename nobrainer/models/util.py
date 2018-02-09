"""Utilities for `nobrainer.models`."""

import tensorflow as tf


def _add_activation_summary(x):
    """Add data from Tensor `x` to histogram."""
    name = x.op.name + '/activations'
    tf.summary.histogram(name, x)
