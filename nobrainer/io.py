"""Input/output methods."""
import csv
import functools
import multiprocessing
import os

import nibabel as nib
import numpy as np
import tensorflow as tf

from .utils import get_num_parallel

_TFRECORDS_FEATURES_DTYPE = "float32"


def read_csv(filepath, skip_header=True, delimiter=","):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]


def read_mapping(filepath, skip_header=True, delimiter=","):
    """Read CSV to dictionary, where first column becomes keys and second
    columns becomes values. Other columns are ignored. Keys and values are
    coerced to integers.
    """
    mapping = read_csv(filepath, skip_header=skip_header, delimiter=delimiter)
    if not all(map(lambda r: len(r) >= 2, mapping)):
        raise ValueError("not all rows in the mapping have at least 2 values")
    try:
        return {int(row[0]): int(row[1]) for row in mapping}
    except ValueError:
        raise ValueError("mapping values must be integers but non-integer encountered")


def read_volume(filepath, dtype=None, return_affine=False, to_ras=False):
    """Return numpy array of data from a neuroimaging file."""
    img = nib.load(filepath)
    if to_ras:
        img = nib.as_closest_canonical(img)
    data = img.get_fdata(caching="unchanged")
    if dtype is not None:
        data = data.astype(dtype)
    return data if not return_affine else (data, img.affine)


def verify_features_labels(
    volume_filepaths,
    volume_shape=(256, 256, 256),
    check_shape=True,
    check_labels_int=True,
    check_labels_gte_zero=True,
    num_parallel_calls=None,
    verbose=1,
):
    """Verify a list of files. This function is meant to be run before
    converting volumes to TFRecords.

    Parameters
    ----------
    volume_filepaths: nested list. Every sublist in the list should contain two
        items: (1) path to feature volume and (2) path to label volume or a scalar.
    volume_shape: tuple of three ints. Shape that both volumes should be.
    check_shape: boolean, if true, validate that the shape of both volumes is
        equal to 'volume_shape'.
    check_labels_int: boolean, if true, validate that every labels volume is an
        integer type or can be safely converted to an integer type.
    check_labels_gte_zero: boolean, if true, validate that every labels volume
        has values greater than or equal to zero.
    num_parallel_calls: int, number of processes to use for multiprocessing. If
        None, will use all available processes.
    verbose: {0, 1, 2}, verbosity of the progress bar. 0 is silent, 1 is verbose,
        and 2 is semi-verbose.

    Returns
    -------
    List of invalid pairs of filepaths. If the list is empty, all filepaths are
    valid.
    """
    from nobrainer.tfrecord import _labels_all_scalar

    for pair in volume_filepaths:
        if len(pair) != 2:
            raise ValueError(
                "all items in 'volume_filepaths' must have length of 2, but"
                " found at least one item with length != 2."
            )

    labels = (y for _, y in volume_filepaths)
    scalar_labels = _labels_all_scalar(labels)

    for pair in volume_filepaths:
        if not os.path.exists(pair[0]):
            raise ValueError("file does not exist: {}".format(pair[0]))
        if not scalar_labels:
            if not os.path.exists(pair[1]):
                raise ValueError("file does not exist: {}".format(pair[1]))

    if scalar_labels:
        map_fn = functools.partial(
            _verify_features_scalar_labels,
            volume_shape=volume_shape,
            check_shape=check_shape,
        )
    else:
        map_fn = functools.partial(
            _verify_features_nonscalar_labels,
            volume_shape=volume_shape,
            check_shape=check_shape,
            check_labels_int=check_labels_int,
            check_labels_gte_zero=check_labels_gte_zero,
        )
    if num_parallel_calls is None:
        num_parallel_calls = get_num_parallel()

    print("Verifying {} examples".format(len(volume_filepaths)))
    progbar = tf.keras.utils.Progbar(len(volume_filepaths), verbose=verbose)
    progbar.update(0)

    outputs = []
    if num_parallel_calls == 1:
        for vf in volume_filepaths:
            valid = map_fn(vf)
            outputs.append(valid)
            progbar.add(1)
    else:
        with multiprocessing.Pool(num_parallel_calls) as p:
            for valid in p.imap(map_fn, volume_filepaths, chunksize=2):
                outputs.append(valid)
                progbar.add(1)
    invalid_files = [
        pair for valid, pair in zip(outputs, volume_filepaths) if not valid
    ]
    return invalid_files


def _verify_features_nonscalar_labels(
    pair_of_paths, *, volume_shape, check_shape, check_labels_int, check_labels_gte_zero
):
    """Verify a pair of features and labels volumes."""
    x = nib.load(pair_of_paths[0])
    y = nib.load(pair_of_paths[1])
    if check_shape:
        if not volume_shape:
            raise ValueError(
                "`volume_shape` must be specified if `check_shape` is true."
            )
        if x.shape != volume_shape:
            return False
        if x.shape != y.shape:
            return False
    if check_labels_int:
        # Quick check of integer type.
        if not np.issubdtype(y.dataobj.dtype, np.integer):
            return False
        y = y.get_fdata(caching="unchanged", dtype=np.float32)
        # Longer check that all values in labels can be cast to int.
        if not np.all(np.mod(y, 1) == 0):
            return False
    if check_labels_gte_zero:
        if not np.all(y >= 0):
            return False
    return True


def _verify_features_scalar_labels(path_scalar, *, volume_shape, check_shape):
    """Check that feature has the desired shape and that label is scalar."""
    from nobrainer.tfrecord import _is_int_or_float

    feature, label = path_scalar
    x = nib.load(feature)
    if check_shape:
        if not volume_shape:
            raise ValueError(
                "`volume_shape` must be specified if `check_shape` is true."
            )
        if x.shape != volume_shape:
            return False
    if not _is_int_or_float(label):
        return False
    return True


def _is_gzipped(filepath):
    """Return True if the file is gzip-compressed, False otherwise."""
    with open(filepath, "rb") as f:
        return f.read(2) == b"\x1f\x8b"
