"""Utilities for `nobrainer.models`."""

import tensorflow as tf

import nobrainer


def _add_activation_summary(x):
    """Add data from Tensor `x` to histogram."""
    name = x.op.name + '/activations'
    tf.summary.histogram(name, x)


def get_estimator(name):
    """Return `tf.estimators.Estimator` subclass of model `name`."""
    estimators = {
        'highres3dnet': nobrainer.models.HighRes3DNet,
        'meshnet': nobrainer.models.MeshNet,
        'quicknat': nobrainer.models.QuickNAT,
    }

    if isinstance(name, tf.estimator.Estimator):
        return name

    try:
        return estimators[name.lower()]
    except KeyError:
        avail = ", ".join(estimators.keys())
        raise ValueError(
            "Model is not available: {}. Available models are {}."
            .format(name, avail)
        )
