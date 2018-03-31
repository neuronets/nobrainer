"""Preprocessing methods."""

import numpy as np


def as_blocks(a, block_shape):
    """Return new array of shape `(n_blocks, *block_shape)`. This separates `a`
    into non-overlapping blocks, each with shape `block_shape`.
    """
    a = np.asarray(a)
    orig_shape = np.asarray(a.shape)

    if a.ndim != 3:
        raise ValueError("This function only works for 3D arrays.")
    if len(block_shape) != 3:
        raise ValueError("block_shape must have three values.")

    blocks = orig_shape // block_shape
    inter_shape = tuple(e for tup in zip(blocks, block_shape) for e in tup)
    new_shape = (-1,) + block_shape
    perm = (0, 2, 4, 1, 3, 5)
    return a.reshape(inter_shape).transpose(perm).reshape(new_shape)


def binarize(a, threshold=0, upper=1, lower=0):
    """Binarize array `a`, where values greater than `threshold` become `upper`
    and all other values become `lower`. Creates new array.
    """
    a = np.asarray(a)
    return np.where(a > threshold, upper, lower)


def from_blocks(a, output_shape):
    """Return new array of shape `output_shape`. This combines non-overlapping
    blocks, likely created by `as_blocks`."""
    a = np.asarray(a)

    if a.ndim != 4:
        raise ValueError("This function only works for 4D arrays.")
    if len(output_shape) != 3:
        raise ValueError("output_shape must have three values.")

    n_blocks = a.shape[0]
    block_shape = a.shape[1:]
    ncbrt = np.cbrt(n_blocks).round(6)
    if not ncbrt.is_integer():
        raise ValueError("Cubed root of number of blocks is not an integer")
    ncbrt = int(ncbrt)
    intershape = (ncbrt, ncbrt, ncbrt, *block_shape)

    return (
        a.reshape(intershape)
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


def preprocess_aparcaseg(a, mapping):
    """Return preprocessed aparc+aseg array. Replaces values in `a` based on
    dictionary `mapping`, and zeros values that are not values in `mapping`.
    """
    a = np.asarray(a)
    a = replace(a, mapping=mapping)
    a_not_in_mapping = ~np.isin(a, mapping.values())
    a[a_not_in_mapping] = 0
    return a


# https://stackoverflow.com/a/47171600
def replace(a, mapping, assume_all_present=False):
    """Replace values in array `a` using dictionary `mapping`.

    Args:
        a: ndarray
        mapping: dict, items in `a` matching a key in `mapping` are replaced
            with the corresponding value. Keys and values may overlap.
        assume_all_present: bool, true if there is key for each unique value in
            `a`. This allows the use of a faster implementation.

    Returns:
        replaced ndarray
    """
    # Extract out keys and values
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks, a)

    if not assume_all_present:
        idx[idx == len(vs)] = 0
        mask = ks[idx] == a
        return np.where(mask, vs[idx], a)
    else:
        return vs[idx]
