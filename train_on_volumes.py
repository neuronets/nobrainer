#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example script to train on neuroimaging volumes."""

import argparse

import numpy as np
import tensorflow as tf

import nobrainer
from nobrainer.io import read_csv, read_mapping
from nobrainer.preprocessing import (
    binarize, preprocess_aparcaseg, normalize_zero_one)
from nobrainer.util import iter_volumes, input_fn_builder

# Data types of labels (x) and features (y).
DT_X = 'float32'
DT_Y = 'int32'
_DT_X_NP = np.dtype(DT_X)
_DT_X_TF = tf.as_dtype(DT_X)
_DT_Y_NP = np.dtype(DT_Y)
_DT_Y_TF = tf.as_dtype(DT_Y)


def train(params):
    """Train estimator."""
    if params['aparcaseg_mapping']:
        tf.logging.info(
            "Reading mapping file: {}".format(params['aparcaseg_mapping']))
        mapping = read_mapping(params['aparcaseg_mapping'])
    else:
        mapping = None

    def normalizer_aparcaseg(features, labels):
        return (
            normalize_zero_one(features),
            preprocess_aparcaseg(labels, mapping))

    def normalizer_brainmask(features, labels):
        return (
            normalize_zero_one(features),
            binarize(labels, threshold=0))

    if params['aparcaseg_mapping'] is not None:
        normalizer = normalizer_aparcaseg
    elif params['brainmask']:
        normalizer = normalizer_brainmask
    else:
        normalizer = None

    list_of_filepaths = read_csv(params['csv'])

    def generator_builder():
        """Return a function that returns a generator."""
        return iter_volumes(
            list_of_filepaths=list_of_filepaths,
            vol_shape=params['vol_shape'],
            block_shape=params['block_shape'],
            x_dtype=_DT_X_NP,
            y_dtype=_DT_Y_NP,
            strides=params['strides'],
            shuffle=True,
            normalizer=normalizer)

    _output_shapes = (
        (*params['block_shape'], 1),
        params['block_shape'])

    input_fn = input_fn_builder(
        generator=generator_builder,
        output_types=(_DT_X_TF, _DT_Y_TF),
        output_shapes=_output_shapes,
        num_epochs=params['n_epochs'],
        batch_size=params['batch_size'],
        # TODO(kaczmarj): add multi-gpu support for training on volumes.
        # multi_gpu=params['multi_gpu'],
        # examples_per_epoch=examples_per_epoch,
    )

    runconfig = tf.estimator.RunConfig(
        save_summary_steps=25,
        save_checkpoints_steps=500,
        keep_checkpoint_max=100)

    model = nobrainer.models.get_estimator(params['model'])(
        n_classes=params['n_classes'],
        optimizer=params['optimizer'],
        learning_rate=params['learning_rate'],
        model_dir=params['model_dir'],
        config=runconfig,
        multi_gpu=params['multi_gpu'])

    # Setup for training and periodic evaluation.
    if params['eval_csv'] is not None:
        eval_list_of_filepaths = read_csv(params['eval_csv'])
        gen = nobrainer.util.iter_volumes(
            list_of_filepaths=eval_list_of_filepaths,
            x_dtype=_DT_X_NP,
            y_dtype=_DT_Y_NP,
            vol_shape=params['vol_shape'],
            block_shape=params['block_shape'],
            strides=params['strides'],
            shuffle=False,
            normalizer=normalizer)

        def _get_eval_features_labels():
            _features = []
            _labels = []
            for _f, _l in gen:
                _features.append(_f)
                _labels.append(_l)
            return np.stack(_features), np.stack(_labels)

        tf.logging.info("Loading evaluation data")
        _eval_features, _eval_labels = _get_eval_features_labels()

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=_eval_features, y=_eval_labels, batch_size=2, num_epochs=1,
            shuffle=False)

        _monitors = [
            tf.contrib.learn.monitors.ValidationMonitor(
                input_fn=eval_input_fn, every_n_steps=2000,
                early_stopping_metric=None, early_stopping_rounds=None)]
        hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
            _monitors, model)

    # Training without evaluation.
    else:
        hooks = None

    model.train(input_fn=input_fn, hooks=hooks)


def _check_required_keys_exist(params):
    keys = {
        'n_classes', 'model', 'model_dir', 'optimizer', 'learning_rate',
        'batch_size', 'vol_shape', 'block_shape', 'brainmask',
        'aparcaseg_mapping', 'csv', 'strides', 'multi_gpu', 'n_epochs',
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
        help="Number of samples per batch. If `--multi-gpu` is specified,"
             " batch is split across available GPUs.")
    p.add_argument(
        '--vol-shape', nargs=3, required=True, type=int,
        help="Height, width, and depth of input data and features.")
    p.add_argument(
        '--block-shape', nargs=3, required=True, type=int,
        help="Height, width, and depth of blocks to take from input data and"
             " features.")
    p.add_argument(
        '--strides', nargs=3, required=True, type=int,
        help="Height, width, and depth of strides Use strides equal to block"
             " shape to train on non-overlapping blocks.")
    p.add_argument(
        '--csv', required=True,
        help="Path to CSV of features, labels for training.")
    p.add_argument(
        '--eval-csv',
        help="Path to CSV of features, labels for periodic evaluation.")
    p.add_argument(
        '-e', '--n-epochs', type=int, default=1,
        help="Number of training epochs")
    p.add_argument(
        '--brainmask', action='store_true',
        help="If specified, binarize labels data")
    p.add_argument(
        '--aparcaseg-mapping',
        help="Path to CSV mapping file. First column contains original labels,"
             " and second column contains new labels in range [0, n_classes-1]"
             ". Header must be included. More than two columns are accepted,"
             " but only the first two columns are used.")
    p.add_argument(
        '--model-dir',
        help="Directory in which to save model checkpoints. If an existing"
             " directory, will resume training from last checkpoint. If not"
             " specified, will use a temporary directory.")
    p.add_argument(
        '--multi-gpu', action='store_true',
        help="If specified, train across all available GPUs. Batches are split"
             " across GPUs.")
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
    params['strides'] = tuple(params['strides'])

    train(params)
