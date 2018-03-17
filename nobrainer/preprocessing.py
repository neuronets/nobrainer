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


def binarize(a, threshold=0, upper=1, lower=0):
    """Binarize array `a`, where values greater than `threshold` become `upper`
    and all other values become `lower`. Creates new array.
    """
    a = np.asarray(a)
    return np.where(a > threshold, upper, lower)


def from_blocks(a, output_shape):
    """Initial implementation of function to go from blocks to full volume."""
    if output_shape != (256, 256, 256) or a.shape != (8, 128, 128, 128):
        raise ValueError(
            "This function only accepts arrays of shape (8, 128, 128, 128) and"
            " output shape of (256, 256, 256). A more general implementation"
            " is in progress.")
    return (
        a.reshape((2, 2, 2, 128, 128, 128))
        .transpose((0, 3, 1, 4, 2, 5))
        .reshape(output_shape))


def normalize_zero_one(a):
    """Return array with values of `a` normalized to range [0, 1].

    This procedure is also known as min-max scaling.
    """
    a = np.asarray(a)
    min_ = a.min()
    return (a - min_) / (a.max() - min_)


def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    return (a - a.mean()) / a.std()


def preprocess_aparcaseg(a, mapping, copy=False):
    """Return preprocessed aparc+aseg array. Replaces values in `a` based on
    diciontary `mapping`, and zeros values that are not values in `mapping`.
    """
    a = replace(a, mapping=mapping, copy=copy)
    max_label = max(mapping.values())
    a[a > max_label] = 0
    return a


def replace(a, mapping, copy=False):
    """Replace values in array `a` with using dictionary `mapping`."""
    a = np.asarray(a)
    # TODO(kaczmarj): this implementation can lead to unexpected behavior if
    # keys and values of mapping overlap.
    if copy:
        a = a.copy()
    for k, v in mapping.items():
        a[a == k] = v
    return a
