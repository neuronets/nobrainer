"""Preprocessing methods."""

import numpy as np


def as_blocks(a, block_shape):
    """Return new array of shape `(n_blocks, *block_shape)`."""
    orig_shape = np.asarray(a.shape)
    blocks = orig_shape // block_shape
    inter_shape = tuple(e for tup in zip(blocks, block_shape) for e in tup)
    new_shape = (-1,) + block_shape
    perm = (0, 2, 4, 1, 3, 5)  # TODO: generalize this.
    return a.reshape(inter_shape).transpose(perm).reshape(new_shape)


def binarize(a, threshold=0, copy=False):
    """Binarize array `a`, where values greater than `threshold` become 1 and
    all other values become 0. Operates in-place unless `copy` is true.
    """
    a = np.asarray(a)
    if copy:
        a = a.copy()
    mask = a > threshold
    a[mask] = 1
    a[~mask] = 0
    return a


def preprocess_aparcaseg(a, mapping):
    """Return preprocessed aparc+aseg array. Replaces values in `a` based on
    diciontary `mapping`, and zeros values that are not values in `mapping`.
    """
    a = replace(a, mapping=mapping)
    max_label = max(mapping.values())
    a[a > max_label] = 0
    return a


def replace(a, mapping, copy=False):
    """Replace values in array `a` with using dictionary `mapping`."""
    a = np.asarray(a)
    if copy:
        a = a.copy()
    for k, v in mapping.items():
        a[a == k] = v
    return a
