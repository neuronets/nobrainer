# -*- coding: utf-8 -*-
"""Utilities."""

# Fuzz factor for stability in computations.
_EPSILON = 1e-07


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
