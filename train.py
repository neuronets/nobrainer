#!/usr/bin/env python3

"""Example training script."""

import argparse
import os

import numpy as np
import tensorflow as tf

import nobrainer

BASE_MODEL_SAVE_PATH = '/om/user/jakubk/nobrainer/models'


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
        previous_params = nobrainer.io.load_json(filepath)
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
        'model', 'model_dir', 'csv', 'learning_rate', 'num_classes',
        'block_shape', 'batch_size',
    }
    for key in keys:
        if key not in params:
            raise ValueError("Required key not in parameters: {}".format(key))


def train(params):
    """Train estimator."""

    print("++ Using parameters:")
    for k, v in params.items():
        print('++', k, v)

    filepaths = nobrainer.io.read_feature_label_filepaths(params['csv'])

    iter_feature_labels = nobrainer.io.iter_features_labels_fn_builder(
        list_of_filepaths=filepaths,
        x_dtype=_DT_X_NP,
        y_dtype=_DT_Y_NP,
        block_shape=params['block_shape'],
        batch_size=params['batch_size'],
    )

    _output_shapes = (
        (None, *params['block_shape'], 1),
        (None, *params['block_shape']),
    )

    input_fn = nobrainer.io.input_fn_builder(
        generator=iter_feature_labels,
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
    p.add_argument('--csv', required=True, type=str)
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
