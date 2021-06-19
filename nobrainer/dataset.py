"""Methods for creating `tf.data.Dataset` objects."""

import glob
import math

import numpy as np
import tensorflow as tf

from .io import _is_gzipped
from .tfrecord import parse_example_fn
from .volume import (
    apply_random_transform,
    apply_random_transform_scalar_labels,
    binarize,
    replace,
    standardize,
    to_blocks,
)

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
    normalizer=standardize,
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
    normalizer: callable, applies this normalization function when creating the
        dataset. to maintain compatibility with prior nobrainer release, this is
        set to standardize by default.
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

    if normalizer is not None:
        # Standard-score the features.
        dataset = dataset.map(lambda x, y: (normalizer(x), y))

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
                return (x, y)

            dataset = dataset.map(_f, num_parallel_calls=num_parallel_calls)
            # This step is necessary because separating into blocks adds a dimension.
            dataset = dataset.unbatch()

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
