#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main command-line interface to nobrainer."""

import argparse
import json
import sys

import tensorflow as tf

from nobrainer.io import read_csv
from nobrainer.models.util import get_estimator
from nobrainer.train import train as _train
from nobrainer.volume import VolumeDataGenerator


def create_parser():
    """Return argument parser for nobrainer training interface."""
    p = argparse.ArgumentParser()

    m = p.add_argument_group('model arguments')

    m.add_argument(
        '-n', '--n-classes', required=True, type=int,
        help="Number of classes to classify")
    m.add_argument(
        '-m', '--model', required=True, choices={'highres3dnet', 'meshnet'},
        help="Model to use")
    m.add_argument(
        '--model-opts', type=json.loads, default={},
        help='JSON string of model-specific options. For example'
             ' `{"n_filters": 71}`.')
    m.add_argument(
        '--model-dir',
        help="Directory in which to save model checkpoints. If an existing"
             " directory, will resume training from last checkpoint. If not"
             " specified, will use a temporary directory.")

    t = p.add_argument_group('train arguments')
    t.add_argument(
        '-o', '--optimizer', required=True,
        help="Optimizer to use for training")
    t.add_argument(
        '-l', '--learning-rate', required=True, type=float,
        help="Learning rate to use with optimizer for training")
    t.add_argument(
        '-b', '--batch-size', required=True, type=int,
        help="Number of samples per batch. If `--multi-gpu` is specified,"
             " batch is split across available GPUs.")
    t.add_argument(
        '-e', '--n-epochs', type=int, default=1,
        help="Number of training epochs")
    t.add_argument(
        '--multi-gpu', action='store_true',
        help="Train across all available GPUs. Batches are split across GPUs.")
    t.add_argument(
        '--prefetch', type=int,
        help="Number of full volumes to prefetch for training and evaluation")
    t.add_argument(
        "--save-summary-steps", type=int, default=25,
        help="Save summaries every this many steps.")
    t.add_argument(
        "--save-checkpoints-steps", type=int, default=100,
        help="Save checkpoints every this many steps.")
    t.add_argument(
        "--keep-checkpoint-max", type=int, default=5,
        help="Maximum number of recent checkpoint files to keep. Use 0 or None"
             " to keep all.")

    d = p.add_argument_group('data arguments')
    d.add_argument(
        '--volume-shape', nargs=3, required=True, type=int,
        help="Height, width, and depth of input data and features.")
    d.add_argument(
        '--block-shape', nargs=3, required=True, type=int,
        help="Height, width, and depth of blocks to take from input data and"
             " features.")
    d.add_argument(
        '--strides', nargs=3, required=True, type=int,
        help="Height, width, and depth of strides Use strides equal to block"
             " shape to train on non-overlapping blocks.")
    d.add_argument(
        '--csv', required=True,
        help="Path to CSV of features, labels for training.")

    s = p.add_argument_group('segmentation task arguments')
    s.add_argument(
        '--binarize', action='store_true',
        help="If specified, binarize labels (e.g., for training a brain"
             " extraction network)")
    s.add_argument(
        '--label-mapping',
        help="Path to CSV mapping file. First column contains original labels,"
             " and second column contains new labels in range [0, n_classes-1]"
             ". Header must be included. More than two columns are accepted,"
             " but only the first two columns are used.")

    e = p.add_argument_group("evaluation arguments")
    e.add_argument(
        '--eval-csv',
        help="Path to CSV of features, labels for periodic evaluation.")

    a = p.add_argument_group('data augmentation arguments')
    a.add_argument('--samplewise-minmax', action='store_true')
    a.add_argument('--samplewise-zscore', action='store_true')
    a.add_argument('--samplewise-center', action='store_true')
    a.add_argument('--samplewise-std-normalization', action='store_true')
    a.add_argument('--rot90-x', action='store_true')
    a.add_argument('--rot90-y', action='store_true')
    a.add_argument('--rot90-z', action='store_true')
    a.add_argument('--rotation-range-x', type=float, default=0.)
    a.add_argument('--rotation-range-y', type=float, default=0.)
    a.add_argument('--rotation-range-z', type=float, default=0.)
    a.add_argument('--shift-range-x', type=float, default=0.)
    a.add_argument('--shift-range-y', type=float, default=0.)
    a.add_argument('--shift-range-z', type=float, default=0.)
    a.add_argument('--flip-x', action='store_true')
    a.add_argument('--flip-y', action='store_true')
    a.add_argument('--flip-z', action='store_true')
    a.add_argument('--brightness-range', type=float, default=0.)
    a.add_argument('--zoom-range', type=float, default=0.)
    a.add_argument('--reduce-contrast', action='store_true')
    a.add_argument('--salt-and-pepper', action='store_true')
    a.add_argument('--gaussian', action='store_true')
    a.add_argument('--rescale', type=float, default=0.)

    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    return parser.parse_args(args)


def train(params):

    model_config = tf.estimator.RunConfig(
        save_summary_steps=params['save_summary_steps'],
        save_checkpoints_steps=params['save_checkpoints_steps'],
        keep_checkpoint_max=params['keep_checkpoint_max'])

    model = get_estimator(params['model'])(
        n_classes=params['n_classes'],
        optimizer=params['optimizer'],
        learning_rate=params['learning_rate'],
        model_dir=params['model_dir'],
        config=model_config,
        multi_gpu=params['multi_gpu'],
        **params['model_opts'])

    label_mapping = None
    if params['label_mapping']:
        tf.logging.info(
            "Reading mapping file: {}".format(params['label_mapping']))
        label_mapping = read_csv(params['label_mapping'])

    filepaths = read_csv(params['csv'])

    volume_data_generator = VolumeDataGenerator(
        samplewise_minmax=params['samplewise_minmax'],
        samplewise_zscore=params['samplewise_zscore'],
        samplewise_center=params['samplewise_center'],
        samplewise_std_normalization=params['samplewise_std_normalization'],
        rot90_x=params['rot90_x'],
        rot90_y=params['rot90_y'],
        rot90_z=params['rot90_z'],
        rotation_range_x=params['rotation_range_x'],
        rotation_range_y=params['rotation_range_y'],
        rotation_range_z=params['rotation_range_z'],
        shift_range_x=params['shift_range_x'],
        shift_range_y=params['shift_range_y'],
        shift_range_z=params['shift_range_z'],
        flip_x=params['flip_x'],
        flip_y=params['flip_y'],
        flip_z=params['flip_z'],
        brightness_range=params['brightness_range'],
        zoom_range=params['zoom_range'],
        reduce_contrast=params['reduce_contrast'],
        salt_and_pepper=params['salt_and_pepper'],
        gaussian=params['gaussian'],
        rescale=params['rescale'],
        binarize_y=params['binarize'],
        mapping_y=label_mapping)

    if params['eval_csv']:
        eval_volume_data_generator = VolumeDataGenerator(
            binarize_y=params['binarize'],
            mapping_y=label_mapping)
    else:
        eval_volume_data_generator = None

    _train(
        model=model,
        volume_data_generator=volume_data_generator,
        filepaths=filepaths,
        volume_shape=params['volume_shape'],
        block_shape=params['block_shape'],
        strides=params['strides'],
        x_dtype='float32',
        y_dtype='int32',
        shuffle=True,
        batch_size=params['batch_size'],
        n_epochs=params['n_epochs'],
        prefetch=params['prefetch'],
        multi_gpu=params['multi_gpu'],
        eval_volume_data_generator=eval_volume_data_generator,
        eval_filepaths=params['eval_csv'])


def main():
    namespace = parse_args(sys.argv[1:])
    params = vars(namespace)

    if params['binarize'] and params['label_mapping']:
        raise ValueError(
            "brainmask and aparcaseg-mapping cannot both be provided.")

    params['block_shape'] = tuple(params['block_shape'])
    params['strides'] = tuple(params['strides'])

    train(params=params)


if __name__ == '__main__':
    main()
