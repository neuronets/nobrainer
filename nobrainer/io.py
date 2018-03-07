""""""

import csv
import json

import nibabel as nib
import numpy as np
import tensorflow as tf

from nobrainer.preprocessing import as_blocks


def read_json(filepath, **kwargs):
    """Load JSON file `filepath` as dictionary. `kwargs` are keyword arguments
    for `json.load()`.
    """
    with open(filepath, 'r') as fp:
        return json.load(fp, **kwargs)


def read_csv(filepath, header=True, delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if header:
            next(reader)  # skip header
        return [row for row in reader]


def save_json(obj, filepath, indent=4, **kwargs):
    """Save `obj` to JSON file `filepath`. `kwargs` are keyword arguments for
    `json.dump()`.
    """
    with open(filepath, 'w') as fp:
        json.dump(obj, fp, indent=indent, **kwargs)
        fp.write('\n')


def load_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data from a neuroimaging file."""
    img = nib.load(filepath)
    data = np.asarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    img.uncache()
    return data if not return_affine else (data, img.affine)


def iterator_nibabel(list_of_filepaths, x_dtype, y_dtype, block_shape,
                     batch_size):
    """Yield tuples of numpy arrays `(features, labels)` from a list of
    filepaths to neuroimaging files.
    """
    for idx, (features_fp, labels_fp) in enumerate(list_of_filepaths):

        tf.logging.info("Reading pair number {}".format(idx))

        try:
            features = load_volume(features_fp, dtype=x_dtype)
            labels = load_volume(labels_fp, dtype=y_dtype)
        except Exception:
            tf.logging.warn(
                "Error reading at least one input file. Skipping {}, {}"
                .format(features_fp, labels_fp)
            )
            continue

        features = as_blocks(features, block_shape)
        features = np.expand_dims(features, -1)

        labels = as_blocks(labels, block_shape)

        n_blocks = features.shape[0]

        if batch_size > n_blocks:
            raise ValueError(
                "Batch size must be less than or equal to the number of"
                " blocks. Got batch size `{}` and {} blocks"
                .format(batch_size, n_blocks)
            )

        iter_range = n_blocks / batch_size
        if not iter_range.is_integer():
            raise ValueError(
                "Batch size must be a factor of number of blocks. Got"
                "batch size `{}` and {} blocks"
                .format(batch_size, n_blocks)
            )

        # Yield non-overlapping batches of blocks.
        for ii in range(int(iter_range)):
            tf.logging.debug("Yielding batch {}".format(ii))
            _start = int(ii * batch_size)
            _end = _start + batch_size
            _slice = slice(_start, _end)
            yield (
                features[_slice, Ellipsis],
                labels[_slice, Ellipsis],
            )


def input_fn_builder(generator, output_types, output_shapes, num_epochs=1,
                     multi_gpu=False, examples_per_epoch=None,
                     batch_size=None):
    """Return `input_fn` handle. `input_fn` returns an instance of
    `tf.estimator.Dataset`, which iterates over `generator`.
    """

    def input_fn():
        """Input function meant to be used with `tf.estimator.Estimator`."""
        dset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=output_types,
            output_shapes=output_shapes,
        )

        dset = dset.repeat(num_epochs)

        if multi_gpu:
            if examples_per_epoch is None or batch_size is None:
                raise ValueError(
                    "`examples_per_epoch` and `batch_size` must be provided"
                    " if using multiple GPUs."
                )
            total_examples = num_epochs * examples_per_epoch
            take_size = batch_size * (total_examples // batch_size)

            print("GOT EXAMPLES PER EPOCH", examples_per_epoch)
            print("GOT TOTAL EXAMPLES", total_examples)
            print("GOT BATCH SIZE", batch_size)
            print("GOT EPOCHS", num_epochs)
            print("TAKING DATASET SIZE", take_size, flush=True)
            dset = dset.take(take_size)

        return dset

    return input_fn
