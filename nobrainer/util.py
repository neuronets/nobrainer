# -*- coding: utf-8 -*-
"""Utilities."""

import numpy as np
import tensorflow as tf


def _shapes_equal(x1, x2):
    """Return whether shapes of arrays or tensors `x1` and `x2` are equal."""
    try:
        x1.shape.as_list() == x2.shape.as_list()  # TensorFlow
    except AttributeError:
        return x1.shape == x2.shape  # NumPy


def _check_shapes_equal(x1, x2):
    """Raise `ValueError` if shapes of arrays or tensors `x1` and `x2` are
    unqeual.
    """
    if not _shapes_equal(x1, x2):
        _shapes = ", ".join((str(x1.shape), str(x2.shape)))
        raise ValueError(
            "Shapes of both arrays or tensors must be equal. Got shapes: "
            + _shapes)


def _check_all_x_in_subset_numpy(x, subset=(0, 1)):
    """Raise `ValueError` if any value of `x` is not in `subset`."""
    x = np.asarray(x)
    masks = tuple(np.equal(x, ii) for ii in subset)
    all_x_in_subset = np.logical_or.reduce(masks).all()
    if not all_x_in_subset:
        _subset = ", ".join(map(str, subset))
        raise ValueError("Not all values are in set {}.".format(_subset))


def input_fn_builder(generator,
                     output_types,
                     output_shapes,
                     num_epochs=1,
                     multi_gpu=False,
                     examples_per_epoch=None,
                     batch_size=1,
                     prefetch=1):
    """Return `input_fn` handle. `input_fn` returns an instance of
    `tf.estimator.Dataset`, which iterates over `generator`.
    """

    def input_fn():
        """Input function meant to be used with `tf.estimator.Estimator`."""
        dset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=output_types,
            output_shapes=output_shapes)

        dset = dset.repeat(num_epochs)

        if multi_gpu:
            if examples_per_epoch is None or batch_size is None:
                raise ValueError(
                    "`examples_per_epoch` and `batch_size` must be provided"
                    " if using multiple GPUs.")
            total_examples = num_epochs * examples_per_epoch
            take_size = batch_size * (total_examples // batch_size)
            tf.logging.info(
                "Total examples (all epochs): {}".format(total_examples))
            tf.logging.info(
                "Training on multiple GPUs. Taking {} samples from dataset."
                .format(take_size))
            dset = dset.take(take_size)

        dset = dset.batch(batch_size)

        if prefetch:
            dset = dset.prefetch(prefetch)

        return dset

    return input_fn


# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py
def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.
    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError(
            'Multi-GPU mode was specified, but no GPUs were found. To use CPU,'
            ' run without --multi-gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = (
            'When running with multiple GPUs, batch size must be a multiple of'
            ' the number of available GPUs. Found {} GPUs with a batch size of'
            ' {}; try --batch-size={} instead.'
            .format(num_gpus, batch_size, batch_size - remainder))
        raise ValueError(err)
