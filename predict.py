#!/usr/bin/env python3
"""Commandline script to predict on T1 volumes."""

import argparse
import sys

import nibabel as nb
import numpy as np
import tensorflow as tf

from nobrainer.models import HighRes3DNet
from nobrainer.preprocessing import as_blocks
from nobrainer.preprocessing import from_blocks
from nobrainer.preprocessing import normalize_zero_one


def predict(features_filepath, batch_size, block_shape):
    """Run `model_obj.predict` on features data in `features_filepath`."""
    x_img = nb.load(features_filepath)

    x = np.asarray(x_img.dataobj).astype(np.float32)
    x = as_blocks(x, block_shape)
    x = normalize_zero_one(x)
    x = x[..., np.newaxis]

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x, shuffle=False, batch_size=batch_size)

    # Load model.
    # TODO(kaczmarj): use tensorflow serving or the export api instead of this.
    # also these shouldn't be hardcoded.
    model = HighRes3DNet(
        n_classes=2,
        model_dir="/opt/nobrainer/trained-models/highres3dnet-brainmask")

    generator = model.predict(input_fn=input_fn)
    predictions = np.zeros(x.shape[:-1])
    for ii, block in enumerate(generator):
        predictions[ii, Ellipsis] = block['class_ids']
    predictions = from_blocks(predictions, x_img.shape)

    return nb.Nifti1Image(
        predictions, affine=x_img.affine, header=x_img.header)


def create_parser():
    """Return argument parser."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input', help="path to T1 on which to predict")
    p.add_argument('output', help="path to save predictions")
    p.add_argument(
        '--batch-size', type=int, default=4,
        help="Batch size. Use lower value if resources are limited")
    p.add_argument(
        '--block-shape', nargs=3, type=int, default=[128, 128, 128],
        help="Separate input volume into 3D blocks of this shape before"
             " training")
    return p


def parse_args(args):
    """Return namespace of arguments."""
    parser = create_parser()
    return parser.parse_args(args)


if __name__ == '__main__':
    namespace = parse_args(sys.argv[1:])
    params = vars(namespace)

    params['block_shape'] = tuple(params['block_shape'])

    predicted_img = predict(
        params['input'],
        batch_size=params['batch_size'],
        block_shape=params['block_shape'])

    nb.save(predicted_img, params['output'])
