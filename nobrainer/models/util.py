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


def get_items_not_in_iterable(items, iterable):
    """Return set of items not in `iterable`."""
    return {i for i in items if i not in iterable}


def check_required_params(required_keys, params):
    """Raise `ValueError` if any items in `required_keys` are not in dictionary
    `params`.
    """
    notin = get_items_not_in_iterable(required_keys, params)
    if notin:
        raise ValueError(
            "required parameters were not found: '{}'."
            .format("', ".join(notin))
        )


def set_default_params(defaults, params):
    """Insert keys and values from `defaults` into dictionary `params` if those
    keys do not exist in `params`. Modifies `params` in-place.
    """
    for k, v in defaults.items():
        params.setdefault(k, v)
