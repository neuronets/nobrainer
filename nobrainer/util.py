"""Utilities."""

import copy
import numbers
import random

import h5py
import numpy as np
import tensorflow as tf

from nobrainer.io import load_volume


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


def create_indices(num_samples, batch_size, shuffle=False):
    """Return list of tuples, where each tuple is """
    import random

    indices = zip(
        range(0, num_samples, batch_size),
        range(batch_size, num_samples + batch_size, batch_size))
    indices = list(indices)

    if shuffle:
        random.shuffle(indices)

    return indices


def _get_n_blocks(arr_shape, kernel_size, strides=1):
    """Return number of blocks that will result from sliding a kernel across
    input array with a certain stride.
    """
    def get_n(a, k, s):
        """Return number of blocks that result from length `a` of input
        array, length `k` of kernel, and stride `s`."""
        return (a - k) / s + 1

    arr_ndim = len(arr_shape)

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * arr_ndim
    if isinstance(strides, numbers.Number):
        strides = [strides] * arr_ndim

    err = "Length of `{}` must be equal to the number of dimensions in `arr`."
    if len(kernel_size) != arr_ndim:
        raise ValueError(err.format('kernel_size'))
    if len(strides) != arr_ndim:
        raise ValueError(err.format('strides'))

    n_blocks = tuple(
        get_n(aa, kk, ss) for aa, kk, ss
        in zip(arr_shape, kernel_size, strides))

    for n in n_blocks:
        if not n.is_integer() or n < 1:
            raise ValueError(
                "Invalid combination of input shape, kernel size, and strides."
                " This combination would create a non-integer or <1 number of"
                " blocks.")

    return tuple(map(int, n_blocks))


def iterblocks_2d(arr, kernel_size, strides=(1, 1)):
    """Yield blocks of 2D array."""
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")

    bx, by = _get_n_blocks(
        arr_shape=arr.shape, kernel_size=kernel_size, strides=strides)

    for ix in range(bx):
        ix *= strides[0]
        for iy in range(by):
            iy *= strides[1]
            ixs = slice(ix, ix + kernel_size[0])
            iys = slice(iy, iy + kernel_size[1])
            yield arr[ixs, iys]


def iterblocks_3d(arr, kernel_size, strides=(1, 1, 1)):
    """Yield blocks of 3D array."""
    if arr.ndim != 3:
        raise ValueError("Input array must be 3D.")

    bx, by, bz = _get_n_blocks(
        arr_shape=arr.shape, kernel_size=kernel_size, strides=strides)

    for ix in range(bx):
        ix *= strides[0]
        for iy in range(by):
            iy *= strides[1]
            for iz in range(bz):
                iz *= strides[2]
                ixs = slice(ix, ix + kernel_size[0])
                iys = slice(iy, iy + kernel_size[1])
                izs = slice(iz, iz + kernel_size[2])
                yield arr[ixs, iys, izs]


def iter_volumes(list_of_filepaths, vol_shape, block_shape, x_dtype, y_dtype,
                 strides=(1, 1, 1), shuffle=False, normalizer=None):
    """Yield tuples of numpy arrays `(features, labels)` from a list of
    filepaths to neuroimaging files.
    """
    list_of_filepaths = copy.deepcopy(list_of_filepaths)
    if shuffle:
        random.shuffle(list_of_filepaths)

    _n_blocks = np.prod(
        _get_n_blocks(
            arr_shape=vol_shape, kernel_size=block_shape, strides=strides))

    tf.logging.info(
        "Will yield {} blocks of shape {} per volume"
        .format(_n_blocks, block_shape))

    for idx, (features_fp, labels_fp) in enumerate(list_of_filepaths):
        try:
            features = load_volume(features_fp, dtype=x_dtype)
            labels = load_volume(labels_fp, dtype=y_dtype)
        except Exception:
            tf.logging.fatal(
                "Error reading at least one input file: {} {}"
                .format(features_fp, labels_fp))
            raise

        if normalizer is not None:
            features, labels = normalizer(features, labels)

        _check_shapes_equal(features, labels)
        feature_gen = iterblocks_3d(
            arr=features, kernel_size=block_shape, strides=strides)
        label_gen = iterblocks_3d(
            arr=labels, kernel_size=block_shape, strides=strides)

        for ff, ll in zip(feature_gen, label_gen):
            yield ff[..., np.newaxis], ll


def iter_hdf5(filepath, x_dataset, y_dataset, x_dtype, y_dtype, shuffle=False,
              normalizer=None):
    """Yield tuples of numpy arrays `(features, labels)` from an HDF5 file."""
    with h5py.File(filepath, 'r') as fp:
        num_x_samples = fp[x_dataset].shape[0]
        num_y_samples = fp[y_dataset].shape[0]

    if num_x_samples != num_y_samples:
        raise ValueError(
            "Number of feature samples is not equal to number of label"
            " samples. Found {x} feature samples and {y} label samples."
            .format(x=num_x_samples, y=num_y_samples))

    indices = list(range(num_x_samples))
    if shuffle:
        random.shuffle(indices)

    for idx in indices:
        with h5py.File(filepath, 'r') as fp:
            features = fp[x_dataset][idx]
            labels = fp[y_dataset][idx]

        features = features[..., np.newaxis]
        features = features.astype(x_dtype)
        labels = labels.astype(y_dtype)

        if normalizer is not None:
            features, labels = normalizer(features, labels)

        yield features, labels


def input_fn_builder(generator, output_types, output_shapes, num_epochs=1,
                     multi_gpu=False, examples_per_epoch=None,
                     batch_size=1):
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
