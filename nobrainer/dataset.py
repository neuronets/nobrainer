"""Methods for creating `tf.data.Dataset` objects."""

import math
import os
from pathlib import Path

import fsspec
import nibabel as nb
import numpy as np
import tensorflow as tf

from .io import _is_gzipped, verify_features_labels
from .tfrecord import _labels_all_scalar, parse_example_fn, write
from .volume import binarize, replace, standardize, to_blocks

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
    parse_fn = parse_example_fn(volume_shape=volume_shape, scalar_label=scalar_label)

    if not shuffle:
        # Determine examples_per_shard from the first TFRecord shard
        # Then set block_length to equal the number of examples per shard
        # so that the interleave method does not inadvertently shuffle data.
        first_shard = (
            dataset.take(1)
            .flat_map(
                lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type)
            )
            .map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
        )
        block_length = len([0 for _ in first_shard])
    else:
        # If the dataset is being shuffled, then we don't care if interleave
        # further shuffles that data even further
        block_length = None

    dataset = dataset.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(
            x, compression_type=compression_type
        ),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=num_parallel_calls,
    )
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
    augment=None,
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
    volume_shape: tuple of at least length 3, the shape of every volume in the TFRecords
        files. Every volume must have the same shape.
    scalar_label: boolean, if `True`, labels are scalars.
    block_shape: tuple of at least length 3, the shape of the non-overlapping sub-volumes
        to take from the full volumes. If None, do not separate the full volumes
        into sub-volumes. Separating into non-overlapping sub-volumes is useful
        (sometimes even necessary) to overcome memory limitations depending on
        the number of model parameters.
    n_epochs: int, number of epochs for the dataset to repeat. If None, the
        dataset will be repeated indefinitely.
    mapping: dict, mapping to replace label values. Values equal to a key in
        the mapping are replaced with the corresponding values in the mapping.
        Values not in `mapping.keys()` are replaced with zeros.
    augment: None, or list of different transforms in the executable sequence
            the corresponding arguments in tuple as e.g.:
            [(addGaussianNoise, {'noise_mean':0.1,'noise_std':0.5}), (...)]
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

    fs, _, _ = fsspec.get_fs_token_paths(file_pattern)
    files = fs.glob(file_pattern)
    if not files:
        raise ValueError("no files found for pattern '{}'".format(file_pattern))

    # Create dataset of all TFRecord files. After this point, the dataset will have
    # two value per iteration: (feature, label).
    shuffle = bool(shuffle_buffer_size)
    compressed = _is_gzipped(files[0], filesys=fs)
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
    if isinstance(augment, bool):
        raise ValueError("Augment no longer supports a boolean expression")

    if augment is not None:
        for transform, kwargs in augment:
            dataset = dataset.map(
                lambda x, y: tf.cond(
                    tf.random.uniform((1,)) > 0.5,
                    true_fn=lambda: transform(x, y, **kwargs),
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
    else:
        if scalar_label:
            dataset = dataset.map(lambda x, y: (x, tf.squeeze(y)))

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

    # If volume_shape is only three dims, add grayscale channel to features.
    # Otherwise, assume that the channels are already in the features.
    if len(volume_shape) == 3:
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


def write_multi_resolution(
    paths,
    tfrecdir=Path(os.getcwd()) / "data",
    resolutions=None,
    shard_size=3,
    n_processes=1,
):
    """Create a multiresolution dataset.

    This returns a dictionary of information that is used by the generative model
    processing class.

    TODO: This function needs to be aligned with the Dataset class
    """
    resolutions = resolutions or [8, 16, 32, 64, 128, 256]
    tfrecdir = Path(tfrecdir)
    tfrecdir.mkdir(exist_ok=True, parents=True)
    template = tfrecdir / "data-train_shard-{shard:03d}.tfrec"

    write(
        features_labels=paths,
        filename_template=str(template),
        examples_per_shard=shard_size,  # change for larger dataset
        multi_resolution=True,
        resolutions=resolutions,
        processes=n_processes,
    )

    datasets = {}
    for resolution in resolutions:
        datasets[resolution] = dict(
            file_pattern=str(tfrecdir / f"*res-{resolution:03d}.tfrec"),
            batch_size=1,
            normalizer=None,
        )
    return datasets


class Dataset:
    """Datasets for training, and validation"""

    def __init__(
        self, n_classes, batch_size, block_shape, volume_shape=None, n_epochs: int = 1
    ):
        self.n_classes = n_classes
        self.volume_shape = volume_shape
        self.block_shape = block_shape
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def from_tfrecords(
        self,
        volume_shape,
        scalar_labels,
        n_volumes,
        template="data/data-train_shard-*.tfrec",
        augment=None,
        shuffle_buffer_size=None,
        num_parallel_calls=1,
    ):
        """Function to retrieve a saved tf record as a Dataset

        template: str, the path to which TFRecord files should be written.
        num_parallel_calls: int, number of processes to use for multiprocessing. If
            None, will use all available processes.
        """
        self.volume_shape = volume_shape

        # replace shard formatting code with * for globbing
        dataset = get_dataset(
            file_pattern=template,
            n_classes=self.n_classes,
            batch_size=self.batch_size,
            volume_shape=self.volume_shape,
            block_shape=self.block_shape,
            n_epochs=self.n_epochs,
            augment=augment,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_calls=num_parallel_calls,
        )
        # Add nobrainer specific attributes
        dataset.scalar_labels = scalar_labels
        dataset.n_volumes = n_volumes
        dataset.volume_shape = self.volume_shape
        return dataset

    def from_files(
        self,
        paths,
        eval_size=0.1,
        tfrecdir=Path(os.getcwd()) / "data",
        shard_size=3,
        augment=None,
        shuffle_buffer_size=None,
        num_parallel_calls=1,
        check_shape=True,
        check_labels_int=False,
        check_labels_gte_zero=False,
    ):
        """Create Nobrainer datasets from data

        template: str, the path to which TFRecord files should be written. A string
            formatting key `shard` should be included to indicate the unique TFRecord file
            when writing to multiple TFRecord files. For example,
            `data_shard-{shard:03d}.tfrec`.
        shard_size: int, number of pairs of `(feature, label)` per TFRecord file.
        check_shape: boolean, if true, validate that the shape of both volumes is
            equal to 'volume_shape'.
        check_labels_int: boolean, if true, validate that every labels volume is an
            integer type or can be safely converted to an integer type.
        check_labels_gte_zero: boolean, if true, validate that every labels volume
            has values greater than or equal to zero.
        num_parallel_calls: int, number of processes to use for multiprocessing. If
            None, will use all available processes.
        """
        # Test that the `filename_template` has a `shard` formatting key.
        template = str(Path(tfrecdir) / "data-{intent}")
        shard_ext = "shard-{shard:03d}.tfrec"

        Neval = np.ceil(len(paths) * eval_size).astype(int)
        Ntrain = len(paths) - Neval

        verify_result = verify_features_labels(
            paths,
            check_shape=check_shape,
            check_labels_int=check_labels_int,
            check_labels_gte_zero=check_labels_gte_zero,
        )
        if len(verify_result) == 0:
            Path(tfrecdir).mkdir(exist_ok=True, parents=True)
            if self.volume_shape is None:
                self.volume_shape = nb.load(paths[0][0]).shape
            write(
                features_labels=paths[:Ntrain],
                filename_template=template.format(intent=f"train_{shard_ext}"),
                examples_per_shard=shard_size,
                processes=num_parallel_calls,
            )
            if Neval > 0:
                write(
                    features_labels=paths[Ntrain:],
                    filename_template=template.format(intent=f"eval_{shard_ext}"),
                    examples_per_shard=shard_size,
                    processes=num_parallel_calls,
                )
            labels = (y for _, y in paths)
            scalar_labels = _labels_all_scalar(labels)
            # replace shard formatting code with * for globbing
            template_train = template.format(intent="train_*.tfrec")
            ds_train = self.from_tfrecords(
                self.volume_shape,
                scalar_labels,
                len(paths[:Ntrain]),
                template=template_train,
                augment=augment,
                shuffle_buffer_size=shuffle_buffer_size,
                num_parallel_calls=num_parallel_calls,
            )
            ds_eval = None
            if Neval > 0:
                template_eval = template.format(intent="eval_*.tfrec")
                ds_eval = self.from_tfrecords(
                    self.volume_shape,
                    scalar_labels,
                    len(paths[Ntrain:]),
                    template=template_eval,
                    augment=None,
                    shuffle_buffer_size=None,
                    num_parallel_calls=num_parallel_calls,
                )
            return ds_train, ds_eval
        raise ValueError(
            "Provided paths did not pass validation. Please "
            "check that they have the same shape, and the "
            "targets have appropriate labels"
        )
