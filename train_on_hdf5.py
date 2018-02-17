#!/usr/bin/env python3

"""Example training script."""

import argparse
import os

import h5py
import numpy as np
import tensorflow as tf

import nobrainer

BASE_MODEL_SAVE_PATH = '/om/user/jakubk/nobrainer/models-hdf5'


def _get_timestamp(fmt='%Y%m%d-%H%M%S'):
    """Return string of current UTC timestamp, formatted with `fmt`."""
    import datetime
    return datetime.datetime.utcnow().strftime(fmt)


def get_model_dir(base_dir, model_name):
    return os.path.join(base_dir, model_name + '_' + _get_timestamp())


def write_params_to_file(filepath, params):
    key = params['model_dir']
    to_save = {key: params}
    if os.path.isfile(filepath):
        previous_params = nobrainer.io.read_json(filepath)
        to_save.update(previous_params)
    nobrainer.io.save_json(to_save, filepath)


# Data types of labels (x) and features (y).
DT_X = 'float32'
DT_Y = 'int32'
_DT_X_NP = np.dtype(DT_X)
_DT_X_TF = tf.as_dtype(DT_X)
_DT_Y_NP = np.dtype(DT_Y)
_DT_Y_TF = tf.as_dtype(DT_Y)


def _check_required_keys_exist(params):
    keys = {
        'model', 'model_dir', 'filepath', 'learning_rate', 'num_classes',
        'block_shape', 'batch_size', 'brainmask',
    }
    for key in keys:
        if key not in params:
            raise ValueError("Required key not in parameters: {}".format(key))


def read_mapping(filepath):
    """Read CSV to dictionary, where first column becomes keys and second columns
    becomes values. Keys and values must be integers.
    """
    mapping = nobrainer.io.read_csv(filepath, header=True)
    mapping = [(int(orig), int(new)) for orig, new, _ in mapping]
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
        num_x_samples, batch_size=batch_size, shuffle=shuffle
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

    tf.logging.info("++ Using parameters:")
    for k, v in params.items():
        tf.logging.info('++ {} : {}'.format(k, v))

    _group = "/{}-iso".format(params['block_shape'][0])
    x_dataset = _group + '/t1'
    y_dataset = _group + '/aparcaseg'

    tf.logging.info(
        'Using features dataset {x} and labels dataset {y}'
        .format(x=x_dataset, y=y_dataset)
    )

    if params['brainmask']:
        mapping = read_mapping(
            '/om2/user/jakubk/openmind-surface-data/data/'
            'FreeSurferColorLUT-mapping-108.csv'
        )
    else:
        mapping = None

    generator = iter_hdf5(
        filepath=params['filepath'],
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
        generator=lambda: generator,
        output_types=(_DT_X_TF, _DT_Y_TF),
        output_shapes=_output_shapes,
    )

    runconfig = tf.estimator.RunConfig(
        save_summary_steps=20, keep_checkpoint_max=20,
    )

    model = nobrainer.models.get_estimator(params['model'])(
        num_classes=params['num_classes'],
        model_dir=params['model_dir'],
        config=runconfig,
        learning_rate=params['learning_rate'],
    )

    write_params_to_file(
        os.path.join(BASE_MODEL_SAVE_PATH, 'directory-mapping.json'), params
    )

    model.train(input_fn=input_fn)


def create_parser():
    """Return argument parser."""
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model', required=True, type=str)
    p.add_argument('-l', '--learning-rate', required=True, type=float)
    p.add_argument('-n', '--num-classes', required=True, type=int)
    p.add_argument('-b', '--batch-size', required=True, type=int)
    p.add_argument('--block-shape', nargs=3, required=True, type=int)
    p.add_argument('--brainmask', action='store_true')
    p.add_argument('--filepath', required=True, type=str)
    p.add_argument('--repeat', type=int, default=0)
    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    return parser.parse_args(args)


if __name__ == '__main__':
    import sys

    namespace = parse_args(sys.argv[1:])
    params = vars(namespace)
    params['block_shape'] = tuple(params['block_shape'])

    params['model_dir'] = get_model_dir(BASE_MODEL_SAVE_PATH, params['model'])

    train(params)
