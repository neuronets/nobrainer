"""Volume processing methods."""

import itertools

import numpy as np
import tensorflow as tf

from nobrainer.transform import get_affine, warp_features_labels


def apply_random_transform(features, labels):
    """Apply a random rigid transformation to `features` and `labels`.

    The same transformation is applied to features and labels. Features are
    interpolated trilinearly, and labels are interpolated with nearest
    neighbors.
    """
    if len(features.shape) != 3 or len(labels.shape) != 3:
        raise ValueError("features and labels must be rank 3")
    if features.shape != labels.shape:
        raise ValueError("shape of features and labels must be the same.")
    # Rotate -180 degrees to 180 degrees in three dimensions.
    rotation = tf.random.uniform(
        shape=[3], minval=-np.pi, maxval=np.pi, dtype=tf.float32
    )

    # Translate at most 5% in any direction, so there's less chance of
    # important data going out of view.
    maxval = 0.05 * features.shape[0]
    translation = tf.random.uniform(
        shape=[3], minval=-maxval, maxval=maxval, dtype=tf.float32
    )

    volume_shape = np.asarray(features.shape)
    matrix = get_affine(
        volume_shape=volume_shape, rotation=rotation, translation=translation
    )
    return warp_features_labels(features=features, labels=labels, matrix=matrix)


def apply_random_transform_scalar_labels(features, labels):
    """Apply a random rigid transformation to `features`.

    Features are interpolated trilinearly, and labels are unchanged because they are
    scalars.
    """
    if len(features.shape) != 3:
        raise ValueError("features must be rank 3")
    if len(labels.shape) != 1:
        raise ValueError("labels must be rank 1")
    # Rotate -180 degrees to 180 degrees in three dimensions.
    rotation = tf.random.uniform(
        shape=[3], minval=-np.pi, maxval=np.pi, dtype=tf.float32
    )

    # Translate at most 5% in any direction, so there's less chance of
    # important data going out of view.
    maxval = 0.05 * features.shape[0]
    translation = tf.random.uniform(
        shape=[3], minval=-maxval, maxval=maxval, dtype=tf.float32
    )

    volume_shape = np.asarray(features.shape)
    matrix = get_affine(
        volume_shape=volume_shape, rotation=rotation, translation=translation
    )
    return warp_features_labels(
        features=features, labels=labels, matrix=matrix, scalar_label=True
    )


def binarize(x):
    """Converts all values greater than 0 to 1 and all others to 0.

    Parameters
    ----------
    x: tensor, values to binarize.

    Returns
    -------
    Tensor of binarized values.
    """
    x = tf.convert_to_tensor(x)
    return tf.cast(x > 0, dtype=x.dtype)


def replace(x, mapping, zero=True):
    """Replace values in tensor `x` using dictionary `mapping`.

    Parameters
    ----------
    x: tensor, values to replace.
    mapping: dict, dictionary mapping original values to new values. Values in
        x equal to a key in the mapping are replaced with the corresponding
        value. Keys and values may overlap.
    zero: boolean, zero values in `x` not in `mapping.keys()`.

    Returns
    -------
    Modified tensor.
    """
    x = tf.cast(x, dtype=tf.int32)
    keys = tf.convert_to_tensor(list(mapping.keys()))
    vals = tf.convert_to_tensor(list(mapping.values()))

    sidx = tf.argsort(keys)
    ks = tf.gather(keys, sidx)
    vs = tf.gather(vals, sidx)

    idx = tf.searchsorted(ks, tf.reshape(x, (-1,)))
    idx = tf.reshape(idx, x.shape)

    # Zero values that are equal to len(vs).
    idx = tf.multiply(idx, tf.cast(tf.not_equal(idx, vs.shape[0]), tf.int32))
    mask = tf.equal(tf.gather(ks, idx), x)
    out = tf.where(mask, tf.gather(vs, idx), x)

    if zero:
        # Zero values in the data array that are not in the mapping values.
        mask = tf.reduce_any(
            tf.equal(tf.expand_dims(out, -1), tf.expand_dims(vals, 0)), -1
        )
        out = tf.multiply(out, tf.cast(mask, tf.int32))

    return out


def standardize(x):
    """Standard score input tensor.

    Implements `(x - mean(x)) / stdev(x)`.

    Parameters
    ----------
    x: tensor, values to standardize.

    Returns
    -------
    Tensor of standardized values. Output has mean 0 and standard deviation 1.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != tf.float32:
        x = tf.cast(x, tf.float32)
    mean, var = tf.nn.moments(x, axes=None)
    std = tf.sqrt(var)
    return (x - mean) / std


def to_blocks(x, block_shape):
    """Split tensor into non-overlapping blocks of shape `block_shape`.

    For the reverse of this function, see `from_blocks`.

    Parameters
    ----------
    x: tensor, 3D tensor to separate into non-overlapping sub-volumes.
    block_shape: tuple of length 3, shape of non-overlapping sub-volumes.

    Returns
    -------
    Tensor with shape `(n_blocks, *block_shape)`.
    """
    x = tf.convert_to_tensor(x)
    volume_shape = np.array(x.shape)

    if isinstance(block_shape, int) == 1:
        block_shape = list(block_shape) * 3
    elif len(block_shape) != 3:
        raise ValueError("expected block_shape to be 1 or 3 values.")

    block_shape = np.asarray(block_shape)
    blocks = volume_shape // block_shape

    inter_shape = list(itertools.chain(*zip(blocks, block_shape)))
    new_shape = (-1, *block_shape)
    perm = (0, 2, 4, 1, 3, 5)  # 3D only
    return tf.reshape(
        tf.transpose(tf.reshape(x, shape=inter_shape), perm=perm), shape=new_shape
    )


def from_blocks(x, output_shape):
    """Combine 4D array of non-overlapping sub-volumes `x` into 3D tensor of
    shape `output_shape`.

    For the reverse of this function, see `to_blocks`.

    Parameters
    ----------
    x: tensor, 4D tensor with shape `(N, *block_shape)`, where `N` is the number
        of sub-volumes.
    output_shape: tuple of length 3, shape of resulting volumes.

    Returns
    -------
    Tensor with shape `output_shape`.
    """
    x = tf.convert_to_tensor(x)
    n_blocks = x.shape[0]
    block_shape = x.shape[1:]
    ncbrt = np.cbrt(n_blocks).round(6)
    if not ncbrt.is_integer():
        raise ValueError("Cubed root of number of blocks is not an integer")
    ncbrt = int(ncbrt)
    intershape = (ncbrt, ncbrt, ncbrt, *block_shape)
    perm = (0, 3, 1, 4, 2, 5)  # 3D only

    return tf.reshape(
        tf.transpose(tf.reshape(x, shape=intershape), perm=perm), shape=output_shape
    )


def adjust_dynamic_range(x, drange_in, drange_out):
    """Scale and shift tensor.

    Implements `(x * scale) + bias`.

    Parameters
    ----------
    x: array, values to scale and shift.
    drange_in: tuple, input range of values
    drange_out: tuple, output range of values

    Returns
    -------
    Array of scaled and shifted values. Output values has range in drange_out.
    """
    x = tf.convert_to_tensor(x)
    scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
    bias = drange_out[0] - drange_in[0] * scale
    return (x * scale) + bias


# Below this line, we implement methods similar to those above but using Numpy.
# This is particularly useful when we use models to predict, because it is
# usually more pleasant to predict on Numpy arrays.


def standardize_numpy(a):
    """Standard score array.

    Implements `(x - mean(x)) / stdev(x)`.

    Parameters
    ----------
    x: array, values to standardize.

    Returns
    -------
    Array of standardized values. Output has mean 0 and standard deviation 1.
    """
    a = np.asarray(a)
    return (a - a.mean()) / a.std()


def normalize_numpy(a):
    """Normalize the array between 0 and 1.

    Implements `(x - min(x)) / (max(x) - min(x))`.

    Parameters
    ----------
    x: array, values to normalize.

    Returns
    -------
    Array of normalized values. Output has min 0 and max 1.
    """
    a = np.asarray(a)
    return (a - a.min()) / (a.max() - a.min())


def from_blocks_numpy(a, output_shape):
    """Combine 4D array of non-overlapping blocks `a` into 3D array of shape
    `output_shape`.

    For the reverse of this function, see `to_blocks_numpy`.

    Parameters
    ----------
    a: array-like, 4D array of blocks with shape (N, *block_shape), where N is
        the number of blocks.
    output_shape: tuple of len 3, shape of the combined array.

    Returns
    -------
    Rank 3 array with shape `output_shape`.
    """
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

    return a.reshape(intershape).transpose((0, 3, 1, 4, 2, 5)).reshape(output_shape)


def to_blocks_numpy(a, block_shape):
    """Return new array of non-overlapping blocks of shape `block_shape` from
    array `a`.

    For the reverse of this function (blocks to array), see `from_blocks_numpy`.

    Parameters
    ----------
    a: array-like, 3D array to block
    block_shape: tuple of len 3, shape of non-overlapping blocks.

    Returns
    -------
    Rank 4 array with shape `(N, *block_shape)`, where N is the number of
    blocks.
    """
    a = np.asarray(a)
    orig_shape = np.asarray(a.shape)

    if a.ndim != 3:
        raise ValueError("This function only supports 3D arrays.")
    if len(block_shape) != 3:
        raise ValueError("block_shape must have three values.")

    blocks = orig_shape // block_shape
    inter_shape = tuple(e for tup in zip(blocks, block_shape) for e in tup)
    new_shape = (-1,) + block_shape
    perm = (0, 2, 4, 1, 3, 5)
    return a.reshape(inter_shape).transpose(perm).reshape(new_shape)


def replace_in_numpy(x, mapping, zero=True):
    """Replace values in numpy ndarray `x` using dictionary `mapping`.

    # Based on https://stackoverflow.com/a/47171600
    """
    # Extract out keys and values
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks, x)

    if not zero:
        idx[idx == len(vs)] = 0
        mask = ks[idx] == x
        return np.where(mask, vs[idx], x)
    else:
        return vs[idx]


def adjust_dynamic_range_numpy(a, drange_in, drange_out):
    """Scale and shift numpy array.

    Implements `(a * scale) + bias`.

    Parameters
    ----------
    a: array, values to scale and shift.
    drange_in: tuple, input range of values
    drange_out: tuple, output range of values

    Returns
    -------
    Array of scaled and shifted values. Output values has range in drange_out.
    """
    a = np.asarray(a)
    scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
    bias = drange_out[0] - drange_in[0] * scale
    return a * scale + bias
