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
from .volume import binarize, replace, to_blocks

AUTOTUNE = tf.data.experimental.AUTOTUNE


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
        self,
        dataset,
        n_volumes,
        volume_shape,
        n_classes,
    ):
        self.dataset = dataset
        self.n_volumes = n_volumes
        self.volume_shape = volume_shape
        self.n_classes = n_classes

    @classmethod
    def from_tfrecords(
        cls,
        file_pattern=None,
        n_volumes=None,
        volume_shape=None,
        block_shape=None,
        scalar_labels=False,
        n_classes=1,
        num_parallel_calls=1,
    ):
        """Function to retrieve a saved tf record as a nobrainer Dataset

        file_pattern: str, the path from which TFRecord files should be read.
        num_parallel_calls: int, number of processes to use for multiprocessing. If
            None, will use all available processes.
        """

        fs, _, _ = fsspec.get_fs_token_paths(file_pattern)
        files = fs.glob(file_pattern)
        if not files:
            raise ValueError("no files found for pattern '{}'".format(file_pattern))

        # Create dataset of all TFRecord files. After this point, the dataset will have
        # two value per iteration: (feature, label).
        compressed = _is_gzipped(files[0], filesys=fs)
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

        # Read each of these files as a TFRecordDataset.
        # Assume all files have same compression type as the first file.
        compression_type = "GZIP" if compressed else None
        cycle_length = 1 if num_parallel_calls is None else num_parallel_calls
        parse_fn = parse_example_fn(
            volume_shape=volume_shape, scalar_labels=scalar_labels
        )

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

        dataset = dataset.interleave(
            map_func=lambda x: tf.data.TFRecordDataset(
                x, compression_type=compression_type
            ),
            cycle_length=cycle_length,
            block_length=block_length,
            num_parallel_calls=num_parallel_calls,
        )
        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
        ds_obj = cls(dataset, n_volumes, volume_shape, n_classes)

        if block_shape:
            ds_obj.block(block_shape)
        if not scalar_labels:
            ds_obj.map_labels()
        # TODO automatically determine batch size
        ds_obj.batch(1)

        return ds_obj

    @classmethod
    def from_files(
        cls,
        filepaths,
        check_shape=True,
        check_labels_int=False,
        check_labels_gte_zero=False,
        out_tfrec_dir=Path(os.getcwd()) / "data",
        shard_size=300,
        num_parallel_calls=1,
        eval_size=0.1,
        n_classes=1,
        block_shape=None,
    ):
        """Create Nobrainer datasets from data
        filepaths: List(str), list of paths to individual input data files.
        check_shape: boolean, if true, validate that the shape of both volumes is
            equal to 'volume_shape'.
        check_labels_int: boolean, if true, validate that every labels volume is an
            integer type or can be safely converted to an integer type.
        check_labels_gte_zero: boolean, if true, validate that every labels volume
            has values greater than or equal to zero.
        out_tfrec_dir: str, the directory to which TFRecord files should be written.
        shard_size: int, number of pairs of `(feature, label)` per TFRecord file.
        num_parallel_calls: int, number of processes to use for multiprocessing. If
            None, will use all available processes.
        eval_size: float, proportion of the input files to reserve for validation.
        n_classes: int, number of output classes
        """
        n_eval = np.ceil(len(filepaths) * eval_size).astype(int)
        n_train = len(filepaths) - n_eval

        verify_result = verify_features_labels(
            filepaths,
            check_shape=check_shape,
            check_labels_int=check_labels_int,
            check_labels_gte_zero=check_labels_gte_zero,
        )
        if len(verify_result) != 0:
            raise ValueError(
                "Provided filepaths did not pass validation. Please "
                "check that they have the same shape, and the "
                "targets have appropriate labels"
            )

        Path(out_tfrec_dir).mkdir(exist_ok=True, parents=True)
        template = str(Path(out_tfrec_dir) / "data-{intent}")
        volume_shape = nb.load(filepaths[0][0]).shape
        write(
            features_labels=filepaths[:n_train],
            filename_template=template.format(intent="train")
            + "_shard-{shard:03d}.tfrec",
            examples_per_shard=shard_size,
            processes=num_parallel_calls,
        )
        if n_eval > 0:
            write(
                features_labels=filepaths[n_train:],
                filename_template=template.format(intent="eval")
                + "_shard-{shard:03d}.tfrec",
                examples_per_shard=shard_size,
                processes=num_parallel_calls,
            )
        labels = (y for _, y in filepaths)
        scalar_labels = _labels_all_scalar(labels)
        # replace shard formatting code with * for globbing
        template_train = template.format(intent="train_*.tfrec")
        ds_train = cls.from_tfrecords(
            template_train,
            n_train,
            volume_shape,
            scalar_labels=scalar_labels,
            n_classes=n_classes,
            block_shape=block_shape,
            num_parallel_calls=num_parallel_calls,
        )
        ds_eval = None
        if n_eval > 0:
            template_eval = template.format(intent="eval_*.tfrec")
            ds_eval = cls.from_tfrecords(
                template_eval,
                n_eval,
                volume_shape,
                scalar_labels=scalar_labels,
                n_classes=n_classes,
                block_shape=block_shape,
                num_parallel_calls=num_parallel_calls,
            )
        return ds_train, ds_eval

    @property
    def batch_size(self):
        return self.dataset.element_spec[0].shape[0]

    @property
    def block_shape(self):
        return tuple(self.dataset.element_spec[0].shape[1:4].as_list())

    @property
    def scalar_labels(self):
        return _labels_all_scalar([y for _, y in self.dataset.as_numpy_iterator()])

    def get_steps_per_epoch(self):
        def get_n(a, k):
            return (a - k) / k + 1

        n_blocks = tuple(
            get_n(aa, kk) for aa, kk in zip(self.volume_shape, self.block_shape)
        )

        for n in n_blocks:
            if not n.is_integer() or n < 1:
                raise ValueError(
                    "cannot create non-overlapping blocks with the given parameters."
                )
        n_blocks_per_volume = np.prod(n_blocks).astype(int)

        steps = n_blocks_per_volume * self.n_volumes / self.batch_size
        steps = math.ceil(steps)
        return steps

    def map(self, func, num_parallel_calls=AUTOTUNE):
        self.dataset = self.dataset.map(func, num_parallel_calls=num_parallel_calls)
        return self

    def normalize(self, normalizer):
        return self.map(lambda x, y: (normalizer(x), y))

    def augment(self, augment_steps, num_parallel_calls=AUTOTUNE):
        batch_size = None
        if len(self.dataset.element_spec[0].shape) > 4:
            batch_size = self.batch_size
            self.dataset = self.dataset.unbatch()

        for transform, kwargs in augment_steps:
            self.map(
                lambda x, y: tf.cond(
                    tf.random.uniform((1,)) > 0.5,
                    true_fn=lambda: transform(x, y, **kwargs),
                    false_fn=lambda: (x, y),
                ),
                num_parallel_calls=num_parallel_calls,
            )

        if batch_size:
            self.batch(batch_size)

        return self

    def block(self, block_shape, num_parallel_calls=AUTOTUNE):
        if not self.scalar_labels:
            self.map(
                lambda x, y: (to_blocks(x, block_shape), to_blocks(y, block_shape)),
                num_parallel_calls=num_parallel_calls,
            )
        else:

            def _f(x, y):
                x = to_blocks(x, block_shape)
                n_blocks = x.shape[0]
                y = tf.repeat(y, n_blocks)
                return (x, y)

            self.map(_f, num_parallel_calls=num_parallel_calls)
        # This step is necessary because separating into blocks adds a dimension.
        self.dataset = self.dataset.unbatch()
        return self

    def map_labels(self, label_mapping=None):
        if self.n_classes < 1:
            raise ValueError("n_classes must be > 0.")

        if label_mapping is not None:
            self.map(lambda x, y: (x, replace(y, label_mapping=label_mapping)))

        if self.n_classes == 1:
            self.map(lambda x, y: (x, tf.expand_dims(binarize(y), -1)))
        elif self.n_classes == 2:
            self.map(lambda x, y: (x, tf.one_hot(binarize(y), self.n_classes)))
        elif self.n_classes > 2:
            self.map(lambda x, y: (x, tf.one_hot(y, self.n_classes)))

        return self

    def batch(self, batch_size):
        # If volume_shape is only three dims, add grayscale channel to features.
        # Otherwise, assume that the channels are already in the features.
        if len(self.dataset.element_spec[0].shape) == 3:
            self.map(lambda x, y: (tf.expand_dims(x, -1), y))
        elif len(self.dataset.element_spec[0].shape) > 4:
            self.dataset = self.dataset.unbatch()

        # Prefetch data to overlap data production with data consumption. The
        # TensorFlow documentation suggests prefetching `batch_size` elements.
        self.dataset = self.dataset.prefetch(buffer_size=batch_size)

        # Batch the dataset, so each iteration gives `batch_size` elements. We drop
        # the remainder so that when training on multiple GPUs, the batch will
        # always be evenly divisible by the number of GPUs. Otherwise, the last
        # batch might have fewer than `batch_size` elements and will cause errors.
        self.dataset = self.dataset.batch(batch_size=batch_size, drop_remainder=True)

        return self

    def shuffle(self, shuffle_buffer_size):
        # Optionally shuffle. We also optionally shuffle the list of files.
        # The TensorFlow recommend shuffling and then repeating.
        self.dataset = self.dataset.shuffle(buffer_size=shuffle_buffer_size)
        return self

    def repeat(self, n_repeats):
        # Repeat the dataset for n_epochs. If n_epochs is None, then repeat
        # indefinitely. If n_epochs is 1, then the dataset will only be iterated
        # through once.
        self.dataset = self.dataset.repeat(n_repeats)
        return self
