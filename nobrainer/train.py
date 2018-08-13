# -*- coding: utf-8 -*-
"""Methods to train models."""

import numpy as np
import tensorflow as tf

from nobrainer.volume import _get_n_blocks

# Data types of features (x) and labels (y).
DT_X = "float32"
DT_Y = "int32"


def train(model,
          volume_data_generator,
          filepaths,
          volume_shape,
          block_shape,
          strides,
          x_dtype=DT_X,
          y_dtype=DT_Y,
          shuffle=True,
          batch_size=8,
          n_epochs=1,
          prefetch=1,
          multi_gpu=False,
          eval_volume_data_generator=None,
          eval_filepaths=None):
    """"""

    input_fn = volume_data_generator.dset_input_fn_builder(
        filepaths=filepaths,
        block_shape=block_shape,
        strides=strides,
        x_dtype=DT_X,
        y_dtype=DT_Y,
        shuffle=shuffle,
        batch_size=batch_size,
        n_epochs=n_epochs,
        prefetch=prefetch,
        multi_gpu=multi_gpu)

    if eval_volume_data_generator is None:
        model.train(input_fn=input_fn)

    else:
        if eval_filepaths is None:
            raise ValueError(
                "must specify `eval_filepaths` if `eval_volume_data_generator`"
                " is used.")
        eval_input_fn = eval_volume_data_generator.dset_input_fn_builder(
            filepaths=eval_filepaths,
            block_shape=block_shape,
            strides=block_shape,  # evaluate on non-overlapping blocks
            x_dtype=DT_X,
            y_dtype=DT_Y,
            shuffle=False,
            batch_size=batch_size,
            n_epochs=n_epochs,
            prefetch=prefetch,
            multi_gpu=multi_gpu)

        examples_per_volume = np.prod(
            _get_n_blocks(
                arr_shape=volume_shape,
                kernel_size=block_shape,
                strides=strides))
        n_samples_per_epoch = examples_per_volume * len(filepaths)
        max_steps = n_samples_per_epoch * n_epochs

        tf.logging.info("Will train for {} steps".format(max_steps))
        train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn,
            max_steps=max_steps,
            hooks=None)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=None,  # Evaluate until input_fn raises end-of-input.
            name=None,
            hooks=None,
            exporters=None,
            start_delay_secs=3600,  # Start evaluating after an hour.
            throttle_secs=3600)  # Evaluate every hour.

        tf.estimator.train_and_evaluate(
            estimator=model,
            train_spec=train_spec,
            eval_spec=eval_spec)
