#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example training script."""

import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import h5py
import numpy as np
import tensorflow as tf

import nobrainer


# Data types of labels (x) and features (y).
DT_X = 'float32'
DT_Y = 'int32'
_DT_X_NP = np.dtype(DT_X)
_DT_X_TF = tf.as_dtype(DT_X)
_DT_Y_NP = np.dtype(DT_Y)
_DT_Y_TF = tf.as_dtype(DT_Y)


def read_mapping(filepath):
    """Read CSV to dictionary, where first column becomes keys and second columns
    becomes values. Keys and values must be integers.
    """
    mapping = nobrainer.io.read_csv(filepath, header=True)
    mapping = [(int(row[0]), int(row[1])) for row in mapping]
    return dict(mapping)


def iter_hdf5(filepath, x_dataset, y_dataset, x_dtype, y_dtype,
              batch_size, shuffle=False, binarize_y=False,
              aparcaseg_mapping=None):
    """Yield tuples of numpy arrays `(features, labels)` from an HDF5 file."""
    with h5py.File(filepath, 'r') as fp:
        num_x_samples = fp[x_dataset].shape[0]
        num_y_samples = fp[y_dataset].shape[0]

    if num_x_samples != num_y_samples:
        raise ValueError(
            "Number of feature samples is not equal to number of label"
            " samples. Found {x} feature samples and {y} label samples."
            .format(x=num_x_samples, y=num_y_samples)
        )

    indices = nobrainer.util.create_indices(
        num_x_samples, batch_size=batch_size, shuffle=shuffle,
    )

    for start, end in indices:

        with h5py.File(filepath, 'r') as fp:
            features = fp[x_dataset][start:end]
            labels = fp[y_dataset][start:end]

        features = np.expand_dims(features, -1)
        features = features.astype(x_dtype)
        labels = labels.astype(y_dtype)

        if aparcaseg_mapping is not None:
            labels = nobrainer.preprocessing.preprocess_aparcaseg(
                labels, aparcaseg_mapping
            )

        if binarize_y:
            labels = nobrainer.preprocessing.binarize(labels, threshold=0)

        yield features, labels


def train(params):
    """Train estimator."""

    x_dataset = params['xdset']
    y_dataset = params['ydset']

    tf.logging.info(
        'Using features dataset {x} and labels dataset {y}'
        .format(x=x_dataset, y=y_dataset)
    )

    with h5py.File(params['hdf5path'], mode='r') as fp:
        examples_per_epoch = fp[x_dataset].shape[0]
        assert examples_per_epoch == fp[y_dataset].shape[0]

    if params['aparcaseg_mapping']:
        tf.logging.info(
            "Reading mapping file: {}".format(params['aparcaseg_mapping'])
        )
        mapping = read_mapping(params['aparcaseg_mapping'])
    else:
        mapping = None

    generator_builder = lambda: iter_hdf5(
        filepath=params['hdf5path'],
        x_dataset=x_dataset,
        y_dataset=y_dataset,
        x_dtype=_DT_X_NP,
        y_dtype=_DT_Y_NP,
        batch_size=params['batch_size'],
        shuffle=True,
        binarize_y=params['brainmask'],  # aparcaseg -> brainmask
        aparcaseg_mapping=mapping,
    )

    _output_shapes = (
        (None, *params['block_shape'], 1),
        (None, *params['block_shape']),
    )

    input_fn = nobrainer.io.input_fn_builder(
        generator=generator_builder,
        output_types=(_DT_X_TF, _DT_Y_TF),
        output_shapes=_output_shapes,
        num_epochs=params['n_epochs'],
        multi_gpu=params['multi_gpu'],
        examples_per_epoch=examples_per_epoch,
        batch_size=params['batch_size'],
    )

    runconfig = tf.estimator.RunConfig(
        save_summary_steps=100, save_checkpoints_steps=100,
        keep_checkpoint_max=25,
    )

    model = nobrainer.models.get_estimator(params['model'])(
        n_classes=params['n_classes'],
        optimizer=params['optimizer'],
        learning_rate=params['learning_rate'],
        model_dir=params['model_dir'],
        config=runconfig,
        multi_gpu=params['multi_gpu'],
    )

    model.train(input_fn=input_fn)


def _check_required_keys_exist(params):
    keys = {
        'n_classes', 'model', 'model_dir', 'optimizer', 'learning_rate',
        'batch_size', 'block_shape', 'brainmask', 'aparcaseg_mapping',
        'hdf5path', 'x_dset', 'y_dset', 'multi_gpu', 'n_epochs',
    }
    for key in keys:
        if key not in params:
            raise ValueError("Required key not in parameters: {}".format(key))


def create_parser():
    """Return argument parser."""
    p = argparse.ArgumentParser()
    p.add_argument(
        '-n', '--n-classes', required=True, type=int,
        help="Number of classes to classify")
    p.add_argument(
        '-m', '--model', required=True, choices={'highres3dnet', 'meshnet'},
        help="Model to use")
    p.add_argument(
        '-o', '--optimizer', required=True,
        help="TensorFlow optimizer to use for training")
    p.add_argument(
        '-l', '--learning-rate', required=True, type=float,
        help="Learning rate to use with optimizer for training")
    p.add_argument(
        '-b', '--batch-size', required=True, type=int,
        help=(
            "Number of samples per batch. If `--multi-gpu` is specified, batch"
            " is split across available GPUs."))
    p.add_argument(
        '--block-shape', nargs=3, required=True, type=int,
        help="Height, width, and depth of input data and features.")
    p.add_argument('--hdf5path', required=True, help="Path to input HDF5")
    p.add_argument('--xdset', required=True, help="Features dataset in HDF5")
    p.add_argument('--ydset', required=True, help="Labels dataset in HDF5")
    p.add_argument(
        '-e', '--n-epochs', type=int, default=1,
        help="Number of training epochs")
    p.add_argument(
        '--brainmask', action='store_true',
        help="If specified, binarize labels data")
    p.add_argument(
        '--aparcaseg-mapping',
        help=(
            "Path to CSV mapping file. First column contains original labels,"
            " and second column contains new labels in range [0, n_classes-1]."
            " Header must be included. More than two columns are accepted, but"
            " only the first two columns are used."))
    p.add_argument(
        '--model-dir',
        help=(
            "Directory in which to save model checkpoints. If an existing"
            " directory, will resume training from last checkpoint. If not"
            " specified, will use a temporary directory."))
    p.add_argument(
        '--multi-gpu', action='store_true',
        help=(
            "If specified, train across all available GPUs. Batches are split"
            " across GPUs."))
    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    return parser.parse_args(args)


if __name__ == '__main__':
    import sys

    namespace = parse_args(sys.argv[1:])
    params = vars(namespace)

    if params['brainmask'] and params['aparcaseg_mapping']:
        raise ValueError(
            "brainmask and aparcaseg-mapping cannot both be provided.")

    params['block_shape'] = tuple(params['block_shape'])

    train(params)
