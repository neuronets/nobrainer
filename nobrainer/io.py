""""""

import csv

import nibabel as nib
import numpy as np
import tensorflow as tf


def read_feature_label_filepaths(filepath, header=True, delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if header:
            next(reader)  # skip header
        return [row for row in reader]


def load_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data from a neuroimaging file."""
    img = nib.load(filepath)
    data = np.asarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    img.uncache()
    return data if not return_affine else (data, img.affine)


def as_blocks(a, block_shape):
    """Return new array of shape `(n_blocks, *block_shape)`."""
    orig_shape = np.asarray(a.shape)
    blocks = orig_shape // block_shape
    inter_shape = tuple(e for tup in zip(blocks, block_shape) for e in tup)
    new_shape = (-1,) + block_shape
    perm = (0, 2, 4, 1, 3, 5)  # TODO: generalize this.
    return a.reshape(inter_shape).transpose(perm).reshape(new_shape)


def iter_features_labels_fn_builder(list_of_filepaths, x_dtype, y_dtype,
                                    block_shape, batch_size):
    """Return function that yields tuples of `(features, labels)`."""

    def iter_features_labels():
        """Yield tuples of numpy arrays `(features, labels)`."""
        for idx, (features_fp, labels_fp) in enumerate(list_of_filepaths):

            tf.logging.info("Reading pair number {}".format(idx))

            features = load_volume(features_fp, dtype=x_dtype)
            features = as_blocks(features, block_shape)
            features = np.expand_dims(features, -1)

            labels = load_volume(labels_fp, dtype=y_dtype)
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

    return iter_features_labels
