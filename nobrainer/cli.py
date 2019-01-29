# -*- coding: utf-8 -*-
"""Main command-line interface to nobrainer."""

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import tensorflow as tf

from nobrainer.io import read_csv
from nobrainer.io import read_mapping
from nobrainer.models.util import get_estimator
from nobrainer.predict import predict as _predict
from nobrainer.validate import validate_from_filepaths
from nobrainer.train import train as _train
from nobrainer.volume import VolumeDataGenerator
from nobrainer.volume import zscore, normalize_zero_one


def create_parser():
    """Return argument parser for nobrainer training interface."""
    p = argparse.ArgumentParser()

    subparsers = p.add_subparsers(
        dest="subparser_name", title="subcommands",
        description="valid subcommands")

    # Training subparser
    tp = subparsers.add_parser('train', help="Train models")

    m = tp.add_argument_group('model arguments')
    m.add_argument(
        '-n', '--n-classes', required=True, type=int,
        help="Number of classes to classify")
    m.add_argument(
        '-m', '--model', required=True, choices={'highres3dnet', 'meshnet', 'meshnetwn', 'meshnetvwn'},
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

    t = tp.add_argument_group('train arguments')
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
        help="Number of blocks to prefetch for training and evaluation")
    t.add_argument(
        '--save-summary-steps', type=int, default=25,
        help="Save summaries every this many steps.")
    t.add_argument(
        '--save-checkpoints-steps', type=int, default=100,
        help="Save checkpoints every this many steps.")
    t.add_argument(
        '--keep-checkpoint-max', type=int, default=5,
        help="Maximum number of recent checkpoint files to keep. Use 0 or None"
             " to keep all.")

    d = tp.add_argument_group('data arguments')
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

    s = tp.add_argument_group('segmentation task arguments')
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

    e = tp.add_argument_group("evaluation arguments")
    e.add_argument(
        '--eval-csv',
        help="Path to CSV of features, labels for periodic evaluation.")

    a = tp.add_argument_group('data augmentation arguments')
    a.add_argument('--samplewise-minmax', action='store_true')
    a.add_argument('--samplewise-zscore', action='store_true')
    a.add_argument('--samplewise-center', action='store_true')
    a.add_argument('--samplewise-std-normalization', action='store_true')
    a.add_argument('--flip', action='store_true')
    a.add_argument('--rescale', type=float, default=0.)
    a.add_argument('--rotate', action='store_true')
    a.add_argument('--gaussian', action='store_true')
    a.add_argument('--reduce-contrast', action='store_true')
    a.add_argument('--salt-and-pepper', action='store_true')
    a.add_argument('--brightness-range', type=float, default=0.)
    a.add_argument('--shift-range', type=float, default=0.)
    a.add_argument('--zoom-range', type=float, default=0.)

    # Prediction subparser
    pp = subparsers.add_parser('predict', help="Predict using SavedModel")
    pp.add_argument('input', help="Filepath to volume on which to predict.")
    pp.add_argument('output', help="Name out output file.")
    ppp = pp.add_argument_group('prediction arguments')
    ppp.add_argument(
        '-b', '--block-shape', nargs=3, required=True, type=int,
        help="Shape of blocks on which predict. Non-overlapping blocks of this"
             " shape are taken from the inputs for prediction.")
    ppp.add_argument(
        '--batch-size', default=4, type=int,
        help="Number of sub-volumes per batch for prediction. Use a smaller"
             " value if memory is insufficient.")
    ppp.add_argument(
        '-m', '--model', required=True, help="Path to saved model.")
    ###
    ppp.add_argument(
        '--n-samples', type=int, default = 1,
        help="Number of sampling.")
    ppp.add_argument('--return-entropy', action='store_true',
        help = 'if you want to return entropy, add this flag.')
    ppp.add_argument('--return-variance', action='store_true',
        help ='if you want to return variance, add this flag.')
    ppp.add_argument('--return-array-from-images', action = 'store_true',
        help = 'if you want to return array instead of image, add this flag.')
    ppp.add_argument('--samplewise-minmax', action='store_true',
        help = 'set normalizer to be minmax. NOTE, normalizer cannot be both minmax and zscore')
    ppp.add_argument('--samplewise-zscore', action='store_true',
        help = 'set normalizer to be zscore. NOTE, normalizer cannot be both minmax and zscore')
    ###


    # Validation subparser
    vp = subparsers.add_parser('validate', help="Validate model using input images and ground truth images.")
    vp.add_argument('--csv', help="Filepath to csv containing image and ground truth on each line.")
    vpp = vp.add_argument_group('validation arguments')
    vpp.add_argument(
        '-b', '--block-shape', nargs=3, required=True, type=int,
        help="Shape of blocks on which predict. Non-overlapping blocks of this"
             " shape are taken from the inputs for prediction.")
    vpp.add_argument(
        '--batch-size', default=4, type=int,
        help="Number of sub-volumes per batch for prediction. Use a smaller"
             " value if memory is insufficient.")
    vpp.add_argument(
        '-m', '--model', required=True, help="Path to saved model.")
    ###
    vpp.add_argument(
        '--n-samples', type=int, default=1,
        help="Number of sampling.")
    vpp.add_argument('--return-entropy', action='store_true',
        help = 'if you want to return entropy, add this flag.')
    vpp.add_argument('--return-variance', action='store_true',
        help ='if you want to return variance, add this flag.')
    vpp.add_argument('--return-array-from-images', action = 'store_true',
        help = 'if you want to return array instead of image, add this flag.')
    vpp.add_argument('--samplewise-minmax', action='store_true',
        help = 'set normalizer to be minmax. NOTE, normalizer cannot be both minmax and zscore')
    vpp.add_argument('--samplewise-zscore', action='store_true',
        help = 'set normalizer to be zscore. NOTE, normalizer cannot be both minmax and zscore')
    vpp.add_argument(
        '--label-mapping',
        help="Path to CSV mapping file. First column contains original labels,"
             " and second column contains new labels in range [0, n_classes-1]"
             ". Header must be included. More than two columns are accepted,"
             " but only the first two columns are used.")
    vpp.add_argument(
        '--output-path',
        help="Path where validation outputs will be saved.")
    vpp.add_argument(
        '-n', '--n-classes', required=True, type=int,
        help="Number of classes the model classifies.")
    ###
    # Save subparser
    sp = subparsers.add_parser('save', help="Save model as SavedModel (.pb)")
    sp.add_argument('savedir', help="Path in which to save SavedModel.")
    spp = sp.add_argument_group('Model arguments')
    spp.add_argument(
        '-m', '--model', required=True,
        help="Nobrainer model (e.g., highres3dnet)")
    spp.add_argument(
        '-d', '--model-dir',
        help="Path to model directory, containing checkpoints, graph, etc.")
    spp.add_argument(
        '-n', '--n-classes', required=True, type=int,
        help="Number of classes the model classifies.")
    spp.add_argument(
        '-b', '--block-shape', nargs=3, required=True, type=int,
        help="Height, width, and depth of data that model takes as input.")
    spp.add_argument(
        '--model-opts', type=json.loads, default={},
        help='JSON string of model-specific options. For example'
             ' `{"n_filters": 71}`.')

    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    namespace = parser.parse_args(args)
    if namespace.subparser_name is None:
        parser.print_usage()
        parser.exit(1)
    return namespace


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
        label_mapping = read_mapping(params['label_mapping'])

    filepaths = read_csv(params['csv'])

    volume_data_generator = VolumeDataGenerator(
        samplewise_minmax=params['samplewise_minmax'],
        samplewise_zscore=params['samplewise_zscore'],
        samplewise_center=params['samplewise_center'],
        samplewise_std_normalization=params['samplewise_std_normalization'],
        flip=params['flip'],
        rescale=params['rescale'],
        rotate=params['rotate'],
        gaussian=params['gaussian'],
        reduce_contrast=params['reduce_contrast'],
        salt_and_pepper=params['salt_and_pepper'],
        brightness_range=params['brightness_range'],
        shift_range=params['shift_range'],
        zoom_range=params['zoom_range'],
        binarize_y=params['binarize'],
        mapping_y=label_mapping)

    if params['eval_csv']:
        eval_filepaths = read_csv(params['eval_csv'])
        eval_volume_data_generator = VolumeDataGenerator(
            binarize_y=params['binarize'],
            mapping_y=label_mapping)
    else:
        eval_filepaths = None
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
        eval_filepaths=eval_filepaths)


def predict(params):
    normalizer = None
    sm = params["samplewise_minmax"]
    sz = params["samplewise_zscore"]
    if sm and sz:
        raise Exception("Normalizer cannot be both minmax and zscore")
    if sm:
        normalizer = normalize_zero_one
    if sz:
        normalizer = zscore

    imgs = _predict(
        inputs=params['input'],
        predictor=params['model'],
        block_shape=params['block_shape'],
        return_variance=params['return_variance'],
        return_entropy=params['return_entropy'],
        return_array_from_images=params['return_array_from_images'],
        normalizer=normalizer,
        n_samples=params['n_samples'],
        batch_size=params['batch_size'])

    outpath = Path(params['output'])
    suffixes = '.'.join(s for s in outpath.suffixes)
    variance_path = outpath.parent / (outpath.stem + '_variance.' + suffixes)
    entropy_path = outpath.parent / (outpath.stem + '_entropy.' + suffixes)

    nib.save(imgs[0], params['output']) # fix
    if not params['return_array_from_images']:
        include_variance = ((params['n_samples'] > 1) and (return_variance))
        include_entropy = ((params['n_samples'] > 1) and (return_entropy))
        if include_variance and return_entropy:
            nib.save(imgs[1], str(variance_path))
            nib.save(imgs[2], str(entropy_path))
        elif include_variance:
            nib.save(imgs[1], str(variance_path))
        elif include_entropy:
            nib.save(imgs[1], str(entropy_path))

def validate(params):
    normalizer = None
    sm = params["samplewise_minmax"]
    sz = params["samplewise_zscore"]
    if sm and sz:
        raise Exception("Normalizer cannot be both minmax and zscore")
    if sm:
        normalizer = normalize_zero_one
    if sz:
        normalizer = zscore
    print(params['model'])
    validate_from_filepaths(
        filepaths=read_csv(params['csv']),
        predictor=params['model'],
        block_shape=params['block_shape'],
        n_classes=params['n_classes'],
        mapping_y=params['label_mapping'],
        output_path=params['output_path'],
        return_variance=params['return_variance'],
        return_entropy=params['return_entropy'],
        return_array_from_images=params['return_array_from_images'],
        n_samples=params['n_samples'],
        normalizer=normalizer,
        batch_size=params['batch_size'])


def save(params):
    volume = tf.placeholder(
        tf.float32, shape=[None, *params['block_shape'], 1])
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={'volume': volume})

    model = get_estimator(params['model'])(
        n_classes=params['n_classes'],
        model_dir=params['model_dir'],
        **params['model_opts'])

    saved_dir = model.export_savedmodel(
        export_dir_base=params['savedir'],
        serving_input_receiver_fn=serving_input_fn)
    print("Saved model to {}".format(saved_dir.decode()))


def main(args=None):
    if args is None:
        namespace = parse_args(sys.argv[1:])
    else:
        namespace = parse_args(args)
    params = vars(namespace)

    if params['subparser_name'] == 'train':
        if params['binarize'] and params['label_mapping']:
            raise ValueError(
                "brainmask and aparcaseg-mapping cannot both be provided.")
        params['block_shape'] = tuple(params['block_shape'])
        params['strides'] = tuple(params['strides'])
        train(params=params)

    elif params['subparser_name'] == 'predict':
        params['block_shape'] = tuple(params['block_shape'])
        if not Path(params['input']).is_file():
            raise FileNotFoundError(
                "file not found: {}".format(params['input']))
        if Path(params['output']).is_file():
            raise FileExistsError(
                "output file exists: {}".format(params['output']))
        predict(params)

    elif params['subparser_name'] == 'save':
        params['block_shape'] = tuple(params['block_shape'])
        if not Path(params['model_dir']).is_dir():
            raise FileExistsError(
                "model directory not found: {}".format(params['modeldir']))
        save(params)

    elif params['subparser_name'] == 'validate':
        params['block_shape'] = tuple(params['block_shape'])
        validate(params)
    else:
        # should never get to this point.
        raise ValueError("invalid subparser.")


# https://stackoverflow.com/a/27674608/5666087
def _exception_handler(exception_type, exception, traceback):
    print("Error [{}]: {}".format(exception_type.__name__, exception))


if __name__ == '__main__':
    sys.excepthook = _exception_handler
    main()
