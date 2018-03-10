"""Utilities."""

import numpy as np


def _shapes_equal(x1, x2):
    """Return whether shapes of arrays or tensors `x1` and `x2` are equal."""
    return x1.shape == x2.shape


def _check_shapes_equal(x1, x2):
    """Raise `ValueError` if shapes of arrays or tensors `x1` and `x2` are
    unqeual.
    """
    if not _shapes_equal(x1, x2):
        _shapes = ", ".join((str(x1.shape), str(x2.shape)))
        raise ValueError(
            "Shapes of both arrays or tensors must be equal. Got shapes: "
            + _shapes)


def _check_all_x_in_subset_numpy(x, subset=(0, 1)):
    """Raise `ValueError` if any value of `x` is not in `subset`."""
    x = np.asarray(x)
    masks = tuple(np.equal(x, ii) for ii in subset)
    all_x_in_subset = np.logical_or.reduce(masks).all()
    if not all_x_in_subset:
        _subset = ", ".join(map(str, subset))
        raise ValueError("Not all values are in set {}.".format(_subset))


def create_indices(num_samples, batch_size, shuffle=False):
    """Return list of tuples, where each tuple is """
    import random

    indices = zip(
        range(0, num_samples, batch_size),
        range(batch_size, num_samples + batch_size, batch_size))
    indices = list(indices)

    if shuffle:
        random.shuffle(indices)

    return indices
