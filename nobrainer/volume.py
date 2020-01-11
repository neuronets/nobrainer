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


def get_dataset(
    file_pattern,
    n_classes,
    batch_size,
    volume_shape,
    block_shape=None,
    n_epochs=None,
    mapping=None,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=None,
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
    labels is `(batch_size, *volume_shape, n_classes)`.
    """

    files = glob.glob(file_pattern)
    if not files:
        raise ValueError("no files found for pattern '{}'".format(file_pattern))

    # Create dataset of all files.
    shuffle = bool(shuffle_buffer_size)
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    # Read each of these files as a TFRecordDataset.
    # Assume all files have same compression type as the first file.
    compression_type = "GZIP" if _is_gzipped(files[0]) else None
    cycle_length = 1 if num_parallel_calls is None else num_parallel_calls
    dataset = dataset.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(
            x, compression_type=compression_type
        ),
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
    )

    # Parse each example in each TFRecords file as a tensor of features and a
    # tensor of labels.
    parse_fn = parse_example_fn(volume_shape=volume_shape, scalar_label=False)
    dataset = dataset.map(map_func=parse_fn)

    # At this point, dataset output will be two tensors, both with shape
    # `volume_shape`. In the next steps, we process these tensors, i.e.,
    # separating them into non-overlapping blocks, binarizing or replacing
    # values in labels, standard-scoring the features, and augmenting.
    preprocess_fn = _get_preprocess_fn(
        n_classes=n_classes, block_shape=block_shape, mapping=mapping, augment=augment
    )
    dataset = dataset.map(preprocess_fn, num_parallel_calls=num_parallel_calls)

    # Flatten the dataset from (n_blocks, *block_shape, 1) to (*block_shape, 1).
    # We do this so that when we batch, we get batches of (batch_size, *block_shape, 1)
    # instead of (batch_size, n_blocks, *block_shape, 1).
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

    # Prefetch data to overlap data production with data consumption. The
    # TensorFlow documentation suggests prefetching `batch_size` elements.
    datset = dataset.prefetch(buffer_size=batch_size)

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

    # 'tensorflow>=1.13'
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
    if x.dtype != tf.float64:
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


def _get_preprocess_fn(n_classes, block_shape=None, mapping=None, augment=False):
    """Creates a `Dataset` of `features` and `labels` preprocessed for binary
    or multiclass segmentation.

    Parameters
    ----------
    n_classes: int, number of classes in output of model, must be greater than 0.
    block_shape: tuple of len 3, if not `None`, the shape of non-overlapping
        sub-volumes to take from the full volumes.
    mapping: dict, dictionary mapping original values in labels to new values.
        Values in x equal to a key in the mapping are replaced with the
        corresponding value. Keys and values may overlap. Only used when
        `n_classes` > 2 (i.e., multiclass segmentation).
    augment: boolean, if true, apply random rigid transformations to the features
        and labels. Features are interpolated trilinearly, and labels are
        interpolated with nearest neighbors. See `apply_random_transform` for
        more information.

    Returns
    -------
    Function that takes float tensors `features` and `labels` and preprocessed
    `features` and `labels`.
    """
    if n_classes < 1:
        raise ValueError("`n_classes` must be at least 1.")

    if n_classes <= 2:

        def preprocess(features, labels):
            if augment:
                # Only apply random augmentation to ~50% of the data.
                features, labels = tf.cond(
                    tf.random.uniform((1,)) > 0.5,
                    true_fn=lambda: apply_random_transform(features, labels),
                    false_fn=lambda: (features, labels),
                )
                # features, labels = apply_random_transform(features, labels)
            features, labels = _preprocess_binary(
                features=features,
                labels=labels,
                n_classes=n_classes,
                block_shape=block_shape,
            )
            return features, labels

    else:

        def preprocess(features, labels):
            if augment:
                # Only apply random augmentation to ~50% of the data.
                features, labels = tf.cond(
                    tf.random.uniform((1,)) > 0.5,
                    true_fn=lambda: apply_random_transform(features, labels),
                    false_fn=lambda: (features, labels),
                )
            features, labels = _preprocess_multiclass(
                features=features,
                labels=labels,
                n_classes=n_classes,
                block_shape=block_shape,
                mapping=mapping,
            )
            return features, labels

    return preprocess


def _preprocess_binary(features, labels, n_classes=1, block_shape=None):
    """Preprocesses `features` and `labels` for binary segmentation.

    Features are standard-scored (mean 0 and standard deviation of 1). Labels
    are binarized (i.e., values above 0 are set to 1). Then, non-overlapping
    blocks of shape `block_shape` are taken from the features and labels. See
    `to_blocks` for more information. Finally, one dimension is added to the
    end of the features tensor as the grascale channel. If `n_classes` is 1,
    then a dimension is added to the labels tensor. If `n_classes` is 2, then
    the labels tensor is transformed into its one-hot encoding.

    Parameters
    ----------
    features: float32 tensor, features volume, must have rank 3.
    labels: float32 tensor, labels volume, must have rank 3.
    n_classes: {1, 2}, number of classes in the output of the target model.
    block_shape: tuple of length 3, shape of sub-blocks to be taken from the
        volumes of features and labels.

    Returns
    -------
    Tuple of preprocessed `features` and `labels`, where `features`
    and `labels` have rank 5: `(n_blocks, x, y, z, c)`.
    """
    x, y = features, labels

    x = tf.convert_to_tensor(x)
    x = standardize(x)
    if block_shape is not None:
        x = to_blocks(x, block_shape=block_shape)
    else:
        x = tf.expand_dims(x, axis=0)
    x = tf.expand_dims(x, axis=-1)  # Add grayscale channel.

    y = tf.convert_to_tensor(y)
    y = binarize(y)
    if block_shape is not None:
        y = to_blocks(y, block_shape=block_shape)
    else:
        y = tf.expand_dims(y, axis=0)
    if n_classes == 1:
        y = tf.expand_dims(y, axis=-1)
    elif n_classes == 2:
        y = tf.one_hot(tf.cast(y, tf.int32), n_classes, dtype=tf.float32)
    else:
        raise ValueError("`n_classes` must be 1 or 2 for binary segmentation.")

    return x, y


def _preprocess_multiclass(features, labels, n_classes, block_shape=None, mapping=None):
    """Preprocesses `features` and `labels` for multiclass segmentation.

    Features are standard-scored (mean 0 and standard deviation of 1). If a
    mapping is provided, values in labels are replaced according to the mapping.
    Values in labels that are not in the keys of the mapping are zeroed. Then,
    non-overlapping blocks of shape `block_shape` are taken from the features
    and labels. See `to_blocks` for more information. Finally, one dimension is
    added to the end of the features tensor as the grascale channel, and the
    labels tensor is transformed into its one-hot encoding.

    Parameters
    ----------
    features: float32 tensor, features volume, must have rank 3.
    labels: float32 tensor, labels volume, must have rank 3.
    n_classes: int, number of classes in the output of the target model, must be
        greater than 2.
    block_shape: tuple of length 3, shape of sub-blocks to be taken from the
        volumes of features and labels.
    mapping: dict, dictionary mapping original values in labels to new values.
        Values in x equal to a key in the mapping are replaced with the
        corresponding value. Keys and values may overlap.

    Returns
    -------
    Tuple of preprocessed `features` and `labels`, where `features`
    have shape ``(n_blocks, x, y, z, 1)` and `labels` have shape
    `(n_blocks, x, y, z, n_classes)`.
    """
    x, y = features, labels

    if n_classes <= 2:
        raise ValueError(
            "`n_classes` must be greater than 2 for multi-class segmentation."
        )

    x = tf.convert_to_tensor(x)
    x = standardize(x)
    if block_shape is not None:
        x = to_blocks(x, block_shape=block_shape)
    else:
        x = tf.expand_dims(x, axis=0)
    x = tf.expand_dims(x, axis=-1)  # Add grayscale channel.

    y = tf.convert_to_tensor(y)
    if mapping is not None:
        y = replace(y, mapping=mapping)
    if block_shape is not None:
        y = to_blocks(y, block_shape=block_shape)
    else:
        y = tf.expand_dims(y, axis=0)

    y = tf.one_hot(tf.cast(y, tf.int32), n_classes, dtype=tf.float32)
    return x, y


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
