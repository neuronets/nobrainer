#!/usr/bin/env python3
"""Example script to train model.

The input CSV must have two columns:
    1. filepaths of features
    2. filepaths of corresponding labels

TODO
----
- Dice coefficient for class 1 (brainmask) is sometimes NaN. This occurs when
    Dice should be zero.
- Input of 1 * 128**3 is too large for 1080ti to train HighRes3DNet. It is OK
    for MeshNet. This issue seems to be related to the `input_fn` used.
"""

import numpy as np
import tensorflow as tf

import nobrainer

PARAMS = {
    'csv': "/om2/user/jakubk/openmind-surface-data/file-lists/master_file_list_brainmask.csv",
    'num_classes': 2,
    'learning_rate': 0.001,
    'data_shape': (256, 256, 256),
    'block_shape': (128, 128, 128),
    'batch_size': 1,
    'repeat': 0,  # number of times to repeat (0 for one epoch)
    'model_dir': './meshnet-today'
}

# Data types of labels (x) and features (y).
DT_X = 'float32'
DT_Y = 'int32'
_DT_X_NP = np.dtype(DT_X)
_DT_X_TF = tf.as_dtype(DT_X)
_DT_Y_NP = np.dtype(DT_Y)
_DT_Y_TF = tf.as_dtype(DT_Y)


def input_fn_builder(generator, output_types, output_shapes, repeat=None):
    """Return input_fn handle."""

    def input_fn():
        """Input function meant to be used with `tf.estimator.Estimator`."""
        dset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=output_types,
            output_shapes=output_shapes,
        ).repeat(repeat)

        return dset

    return input_fn


def main():
    """Train estimator."""

    filepaths = nobrainer.io.read_feature_label_filepaths(PARAMS['csv'])

    iter_feature_labels = nobrainer.io.iter_features_labels_fn_builder(
        list_of_filepaths=filepaths,
        x_dtype=_DT_X_NP,
        y_dtype=_DT_Y_NP,
        block_shape=PARAMS['block_shape'],
        batch_size=PARAMS['batch_size'],
    )

    _output_shapes = (
        (None, *PARAMS['block_shape'], 1),
        (None, *PARAMS['block_shape']),
    )

    input_fn = input_fn_builder(
        generator=iter_feature_labels,
        output_types=(_DT_X_TF, _DT_Y_TF),
        output_shapes=_output_shapes,
    )

    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True

    with tf.Session(config=_config):

        runconfig = tf.estimator.RunConfig(
            save_summary_steps=20, keep_checkpoint_max=20,
        )

        model = nobrainer.models.MeshNet(
            num_classes=PARAMS['num_classes'],
            model_dir=PARAMS['model_dir'],
            config=runconfig,
            learning_rate=PARAMS['learning_rate'],
        )

        model.train(input_fn=input_fn)


if __name__ == '__main__':
    main()
