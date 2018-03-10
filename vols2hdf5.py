#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to convert neuroimaging volumes to volumes."""

import argparse
import glob
import logging
from multiprocessing import Pool
import os
import sys
import time
import types
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import h5py
import numpy as np
import pandas as pd

from nobrainer.io import load_volume, read_csv
from nobrainer.preprocessing import as_blocks
from nobrainer.util import _check_shapes_equal


def get_logger():
    logger = logging.getLogger('vols2hdf5')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger()


def group(iterable, chunksize=10):
    """Yield chunks of `iterable`. Does not support generators."""
    if isinstance(iterable, types.GeneratorType):
        raise TypeError("This function does not support generators.")

    for i in range(0, len(iterable), chunksize):
        yield iterable[i:i + chunksize]


def _preprocess_one(path, block_shape):
    return as_blocks(load_volume(path), block_shape)


def _preprocess_one_multiproc(path_block_shape):
    """Iterable of length 2 (filepath, blockshape)."""
    path, block_shape = path_block_shape
    return _preprocess_one(path, block_shape)


def remove_empty_slices_(mask_arr, other_arr, axis=(1, 2, 3)):
    """Return `mask_arr` and `other_arr` with empty blocks in `mask_arr`
    removed.
    """
    _check_shapes_equal(mask_arr, other_arr)
    mask = mask_arr.any(axis=axis)
    return mask_arr[mask, ...], other_arr[mask, ...]


def get_list_of_t1_aparcaseg(subjectsdir):
    """Return dataframe where rows contain unique pairs of T1 and aparcaseg."""
    features_pattern = os.path.join(subjectsdir, '**', 'mri', 'T1.mgz')
    labels_pattern = os.path.join(subjectsdir, '**', 'mri', 'aparc+aseg.mgz')

    features = glob.glob(features_pattern)
    features = pd.Series(
        features, index=[os.path.dirname(x) for x in features])
    features.name = 'features'

    labels = glob.glob(labels_pattern)
    labels = pd.Series(
        labels, index=[os.path.dirname(x) for x in labels])
    labels.name = 'labels'

    return pd.concat([features, labels], axis=1).values.tolist()


class HdfSinker:
    """Sink data to HDF5 file.

    Args:
        path: str, path-like. Path to hdf5 file.
        datasets: iterable of dataset names.
        dtypes: iterable of dtypes, one per dataset.
        shapes: iterable of tuples of shapes, one per dataset.
        overwrite: boolean, if true, overwrite file if it exists.
    """
    def __init__(self, path, overwrite=False):
        self.path = path

        if overwrite:
            with h5py.File(self.path, mode='w'):
                pass

    def create_dataset(self, *args, **kwds):
        """Create HDF5 dataset. See documentation for
        `h5py.File.create_dataset`.
        """
        with h5py.File(self.path, mode='a') as fp:
            fp.create_dataset(*args, **kwds)

    def append(self, data, dataset):
        """Append array `data` to HDF5 dataset. Return indices of appended
        items.
        """
        with h5py.File(self.path, mode='a') as fp:
            dset = fp[dataset]
            dset_original_len = dset.shape[0]
            data_len = data.shape[0]
            dset_new_len = dset_original_len + data_len

            dset.resize(dset_new_len, axis=0)
            dset[-data_len:] = data

            return (dset_original_len, dset_new_len)


def _create_datasets(sinker, group_name, block_shape, features_dtype,
                     labels_dtype, compression=None, compression_opts=None):
    """Create two datasets in `sinker` under group `group_name`, one for
    features and thee other for labels.

    Returns:
        None
    """
    sinker.create_dataset(
        group_name + '/features', shape=(0, *block_shape),
        dtype=features_dtype, maxshape=(None, *block_shape),
        compression=compression, compression_opts=compression_opts)

    sinker.create_dataset(
        group_name + "/labels", shape=(0, *block_shape), dtype=labels_dtype,
        maxshape=(None, *block_shape), compression=compression,
        compression_opts=compression_opts)


def vols2hdf5_one_group(sinker, list_of_files, group_name, block_shape,
                        remove_empty_slices=False, chunksize=10, n_cpu=1):
    """Convert the volumes in `list_of_files` to HDF5.

    Args:
        sinker: HdfSinker instance
        list_of_files: list of tuples, where each tuple has two items
            `(feature_filepath, label_filepath)`. The data in features are
            placed into the `features` dataset, and the data in `labels` are
            place into `labels` dataset.
        group_name: name of group to create dataset within. Two datasets are
            created within this group: `features` and `labels`.
        block_shape: tuple, shape of blocks into which volumes are separated.
            At the moment, this produces non-overlapping blocks.

    Returns:
        None
    """
    n_pairs = len(list_of_files)
    total_iters = int(np.ceil(n_pairs / chunksize))

    logger.info(
        "Total iterations for this block shape: {}".format(total_iters))

    gen = group(list_of_files[:n_pairs], chunksize=chunksize)

    for idx, this_buffer in enumerate(gen):

        logger.info("Iteration {} of {}".format(idx + 1, total_iters))

        # Load volumes in parallel if multiple CPUs specified.
        if n_cpu > 1:
            logger.debug("Loading features with {} CPUs".format(n_cpu))
            with Pool(processes=n_cpu) as pool:
                these_features = np.concatenate(
                    pool.map(
                        _preprocess_one_multiproc,
                        [(x, block_shape) for x, _ in this_buffer]))
            logger.debug("Loading labels with {} CPUs".format(n_cpu))
            with Pool(processes=n_cpu) as pool:
                these_labels = np.concatenate(
                    pool.map(
                        _preprocess_one_multiproc,
                        [(y, block_shape) for _, y in this_buffer]))
        else:
            logger.debug("Loading features")
            these_features = np.concatenate(
                tuple(_preprocess_one(x, block_shape) for x, _ in this_buffer))
            logger.debug("Loading labels")
            these_labels = np.concatenate(
                tuple(_preprocess_one(y, block_shape) for _, y in this_buffer))

        logger.debug("Features have shape {}".format(these_features.shape))
        logger.debug("Labels have shape {}".format(these_labels.shape))

        # Remove empty (i.e., all zero) labels blocks and their corresponding
        # features blocks.
        if remove_empty_slices:
            logger.debug("Removing empty blocks")
            these_labels, these_features = remove_empty_slices_(
                mask_arr=these_labels, other_arr=these_features)

        logger.debug(
            "Appending features to {}".format(group_name + '/features'))
        _, _end = sinker.append(these_features, group_name + '/features')
        logger.debug(
            "Dataset {} now has {} samples"
            .format(group_name + '/features', _end))

        logger.debug("Appending labels to {}".format(group_name + '/labels'))
        _, _end = sinker.append(these_labels, group_name + '/labels')
        logger.debug(
            "Dataset {} now has {} samples"
            .format(group_name + '/labels', _end))

        del these_features
        del these_labels


def create_parser():
    """Return argument parser."""
    p = argparse.ArgumentParser()
    h = (
        "Path to CSV or value of FreeSurfer $SUBJECTS_DIR. If CSV, each row"
        " must have two items: `filepath_to_feature_vol,filepath_to_labels_vol"
        "`. CSV must also have a header, although the contents do not matter."
        " If value of FreeSurfer $SUBJECTS_DIR, directory must exist.")
    p.add_argument('input', nargs='?', help=h)
    h = ("Path to saved HDF5 file.")
    p.add_argument('-o', '--outfile', required=True, help=h)
    p.add_argument(
        '--block-shape', required=True, nargs=3, type=int, action='append')
    p.add_argument('-fdt', '--features-dtype', default="float32")
    p.add_argument('-ldt', '--labels-dtype', default='int32')
    h = ("If true, remove empty blocks in labels and their corresponding"
         " feature blocks.")
    p.add_argument('--rm-empty', action='store_true', help=h)
    h = "Number of pairs of volumes to load and append to HDF5 at a time."
    p.add_argument('--chunksize', type=int, default=10, help=h)
    h = "Number of CPU processes to use when loading volumes."
    p.add_argument('-N', '--ncpu', type=int, default=1, help=h)
    h = "If true, overwrite HDF5 outfile if it already exists."
    p.add_argument('--overwrite', action='store_true', help=h)
    h = "Compression to use."
    p.add_argument(
        '--compression', default='gzip', choices={'gzip', 'lfz', 'szip'},
        help="Compression to use.")
    p.add_argument(
        '--compression-opts', default=1, type=int, help="Compression options.")
    p.add_argument('--save-filepaths', help="path to save CSV of filepaths.")
    p.add_argument('-v', '--verbose', action='count', default=0)
    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    return parser.parse_args(args)


if __name__ == '__main__':

    time_zero = time.time()
    namespace = parse_args(sys.argv[1:])
    params = vars(namespace)

    if params['verbose'] >= 1:
        logger.setLevel(logging.DEBUG)
    elif params['verbose'] == 0:
        logger.setLevel(logging.INFO)

    if os.path.isdir(params['input']):
        logger.info("Assuming SUBJECTS_DIR was passed in. Findings file pairs")
        list_of_files = get_list_of_t1_aparcaseg(params['input'])

    elif os.path.isfile(params['input']):
        logger.info("Reading CSV")
        list_of_files = read_csv(params['input'])
    else:
        raise ValueError(
            "Input must be the path to an existing FreeSurfer SUBJECTS_DIR or"
            " to an existing CSV file.")

    logger.info("Found {} pairs of volumes".format(len(list_of_files)))
    logger.info("User requested chunk size of {}".format(params['chunksize']))
    logger.info(
        "Will iterate over {} set(s) of block shape(s)"
        .format(len(params['block_shape'])))

    if params['save_filepaths'] is not None:
        _df = pd.DataFrame(list_of_files)
        _df.columns = ["features", "labels"]
        logger.info(
            "Saving CSV of filepaths found by this script to {}"
            .format(params['save_filepaths']))
        _df.to_csv(params['save_filepaths'], index=False)
        del _df

    sinker = HdfSinker(path=params['outfile'], overwrite=params['overwrite'])
    logger.info("Will save data to {}".format(params['outfile']))

    # Iterate over the requested block shapes.
    for block_shape in params['block_shape']:  # list of block shapes
        block_shape = tuple(block_shape)
        group_n = "/" + "x".join(map(str, block_shape))
        logger.info(
            "Iteratively appending to features and labels datasets in group"
            " {}".format(group_n))
        _create_datasets(
            sinker=sinker,
            group_name=group_n,
            block_shape=block_shape,
            features_dtype=params['features_dtype'],
            labels_dtype=params['labels_dtype'],
            compression=params['compression'],
            compression_opts=params['compression_opts'])

        vols2hdf5_one_group(
            sinker=sinker,
            list_of_files=list_of_files,
            group_name=group_n,
            block_shape=block_shape,
            remove_empty_slices=params['rm_empty'],
            chunksize=params['chunksize'],
            n_cpu=params['ncpu'],
        )

    time_elapsed = time.time() - time_zero
    logger.info("Elapsed time: {} seconds".format(int(time_elapsed)))
    logger.info("Finished.")
