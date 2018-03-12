"""Utilities for `nobrainer.models`."""

import tensorflow as tf

import nobrainer


def get_estimator(name):
    """Return `tf.estimators.Estimator` subclass of model `name`. If `name`
    is an Estimator object, return the object.
    """
    try:
        if issubclass(name, tf.estimator.Estimator):
            return name
    except TypeError:
        pass

    estimators = {
        'highres3dnet': nobrainer.models.highres3dnet.HighRes3DNet,
        'meshnet': nobrainer.models.meshnet.MeshNet,
        'quicknat': nobrainer.models.quicknat.QuickNAT,
    }

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


def check_optimizer_for_training(optimizer, mode):
    """Raise `ValueError` if `optimizer` is None when training."""
    if mode == tf.estimator.ModeKeys.TRAIN and optimizer is None:
        raise ValueError("Optimizer must be provided when training.")
