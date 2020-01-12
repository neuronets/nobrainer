"""Volume processing methods."""

import glob
import itertools
import math

import numpy as np
import tensorflow as tf

from nobrainer.io import _is_gzipped
from nobrainer.tfrecord import parse_example_fn
from nobrainer.transform import get_affine
from nobrainer.transform import warp_features_labels

AUTOTUNE = tf.data.experimental.AUTOTUNE


def tfrecord_dataset(
    file_pattern,
    volume_shape,
    shuffle,
    scalar_label,
    compressed=True,
    num_parallel_calls=AUTOTUNE,
):
    """Return `tf.data.Dataset` from TFRecord files."""
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    # Read each of these files as a TFRecordDataset.
    # Assume all files have same compression type as the first file.
    compression_type = "GZIP" if compressed else None
    cycle_length = 1 if num_parallel_calls is None else num_parallel_calls
    dataset = dataset.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(
            x, compression_type=compression_type
        ),
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
    )
    parse_fn = parse_example_fn(volume_shape=volume_shape, scalar_label=scalar_label)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
    return dataset


def get_dataset(
    file_pattern,
    n_classes,
    batch_size,
    volume_shape,
    scalar_label=False,
    block_shape=None,
    n_epochs=None,
    mapping=None,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=AUTOTUNE,
):
    """Return `tf.data.Dataset` that preprocesses data for training or prediction.

    Labels are preprocessed for binary or multiclass segmentation according to
    `n_classes`.

    Parameters
    ----------
    file_pattern: str, expression that can be globbed to get TFRecords files
        for this dataset. For example 'data/training_*.tfrecords'.
    n_classes: int, number of classes to segment. Values of 1 and 2 indicate
        binary segmentation (foreground vs background), and values greater than
        2 indicate multiclass segmentation.
    batch_size: int, number of elements per batch.
    volume_shape: tuple of length 3, the shape of every volume in the TFRecords
        files. Every volume must have the same shape.
    scalar_label: boolean, if `True`, labels are scalars.
    block_shape: tuple of length 3, the shape of the non-overlapping sub-volumes
        to take from the full volumes. If None, do not separate the full volumes
        into sub-volumes. Separating into non-overlapping sub-volumes is useful
        (sometimes even necessary) to overcome memory limitations depending on
        the number of model parameters.
    n_epochs: int, number of epochs for the dataset to repeat. If None, the
        dataset will be repeated indefinitely.
    mapping: dict, mapping to replace label values. Values equal to a key in
        the mapping are replaced with the corresponding values in the mapping.
        Values not in `mapping.keys()` are replaced with zeros.
    augment: boolean, if true, apply random rigid transformations to the
        features and labels. The rigid transformations are applied to the full
        volumes.
    shuffle_buffer_size: int, buffer of full volumes to shuffle. If this is not
        None, then the list of files found by 'file_pattern' is also shuffled
        at every iteration.
    num_parallel_calls: int, number of parallel calls to make for data loading
        and processing.

    Returns
    -------
    `tf.data.Dataset` of features and labels. If block_shape is not None, the
    shape of features is `(batch_size, *block_shape, 1)` and the shape of labels
    is `(batch_size, *block_shape, n_classes)`. If block_shape is None, then
    the shape of features is `(batch_size, *volume_shape, 1)` and the shape of
    labels is `(batch_size, *volume_shape, n_classes)`. If `scalar_label` is `True,
    the shape of labels is always `(batch_size,)`.
    """

    files = glob.glob(file_pattern)
    if not files:
        raise ValueError("no files found for pattern '{}'".format(file_pattern))

    # Create dataset of all TFRecord files. After this point, the dataset will have
    # two value per iteration: (feature, label).
    shuffle = bool(shuffle_buffer_size)
    compressed = _is_gzipped(files[0])
    dataset = tfrecord_dataset(
        file_pattern=file_pattern,
        volume_shape=volume_shape,
        shuffle=shuffle,
        scalar_label=scalar_label,
        compressed=compressed,
        num_parallel_calls=num_parallel_calls,
    )

    # Standard-score the features.
    dataset = dataset.map(lambda x, y: (standardize(x), y))

    # Separate into blocks, if requested.
    if block_shape is not None:
        if not scalar_label:
            dataset = dataset.map(
                lambda x, y: (to_blocks(x, block_shape), to_blocks(y, block_shape)),
                num_parallel_calls=num_parallel_calls,
            )
            # This step is necessary because separating into blocks adds a dimension.
            dataset = dataset.unbatch()
        if scalar_label:

            def _f(x, y):
                x = to_blocks(x, block_shape)
                n_blocks = x.shape[0]
                y = tf.repeat(y, n_blocks)

            dataset = dataset.map(_f, num_parallel_calls=num_parallel_calls)
            # This step is necessary because separating into blocks adds a dimension.
            dataset = dataset.unbatch()

    # Augment examples if requested.
    if augment:
        if not scalar_label:
            dataset = dataset.map(
                lambda x, y: tf.cond(
                    tf.random.uniform((1,)) > 0.5,
                    true_fn=lambda: apply_random_transform(x, y),
                    false_fn=lambda: (x, y),
                ),
                num_parallel_calls=num_parallel_calls,
            )
        else:
            dataset = dataset.map(
                lambda x, y: tf.cond(
                    tf.random.uniform((1,)) > 0.5,
                    true_fn=lambda: apply_random_transform_scalar_labels(x, y),
                    false_fn=lambda: (x, y),
                ),
                num_parallel_calls=num_parallel_calls,
            )

    # Binarize or replace labels according to mapping.
    if not scalar_label:
        if n_classes < 1:
            raise ValueError("n_classes must be > 0.")
        elif n_classes == 1:
            dataset = dataset.map(lambda x, y: (x, tf.expand_dims(binarize(y), -1)))
        elif n_classes == 2:
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(binarize(y), n_classes)))
        elif n_classes > 2:
            if mapping is not None:
                dataset = dataset.map(lambda x, y: (x, replace(y, mapping=mapping)))
            dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, n_classes)))

    # Add grayscale channel to features.
    # TODO: in the future, multi-channel features should be supported.
    dataset = dataset.map(lambda x, y: (tf.expand_dims(x, -1), y))

    # Prefetch data to overlap data production with data consumption. The
    # TensorFlow documentation suggests prefetching `batch_size` elements.
    dataset = dataset.prefetch(buffer_size=batch_size)

    # Batch the dataset, so each iteration gives `batch_size` elements. We drop
    # the remainder so that when training on multiple GPUs, the batch will
    # always be evenly divisible by the number of GPUs. Otherwise, the last
    # batch might have fewer than `batch_size` elements and will cause errors.
    if batch_size is not None:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # Optionally shuffle. We also optionally shuffle the list of files.
    # The TensorFlow recommend shuffling and then repeating.
    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat the dataset for n_epochs. If n_epochs is None, then repeat
    # indefinitely. If n_epochs is 1, then the dataset will only be iterated
    # through once.
    dataset = dataset.repeat(n_epochs)

    return dataset


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
    if len(features.shape) != 1:
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


# numpy implementation https://stackoverflow.com/a/47171600
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


def get_steps_per_epoch(n_volumes, volume_shape, block_shape, batch_size):
    def get_n(a, k):
        return (a - k) / k + 1

    n_blocks = tuple(get_n(aa, kk) for aa, kk in zip(volume_shape, block_shape))

    for n in n_blocks:
        if not n.is_integer() or n < 1:
            raise ValueError(
                "cannot create non-overlapping blocks with the given parameters."
            )
    n_blocks_per_volume = np.prod(n_blocks).astype(int)

    steps = n_blocks_per_volume * n_volumes / batch_size
    steps = math.ceil(steps)
    return steps


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
