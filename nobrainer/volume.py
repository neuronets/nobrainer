# -*- coding: utf-8 -*-
"""Methods for real-time augmentation of volumetric data.

Based on `keras.preprocessing.image`.
"""

import copy
import numbers
import random

import numpy as np
from scipy import ndimage
import tensorflow as tf

from nobrainer.io import read_volume
from nobrainer.util import _check_shapes_equal


def binarize(a, threshold=0, upper=1, lower=0):
    """Binarize array `a`, where values greater than `threshold` become `upper`
    and all other values become `lower`. Creates new array.
    """
    a = np.asarray(a)
    return np.where(a > threshold, upper, lower)


def change_brightness(x, c, lower_clip=0., upper_clip=1., out=None):
    """Change the brightness by adding `c`.

    Args:
        x: array-like, data to modify
        c: numeric, value to add to data
        lower_clip: numeric, lower bound to clip data
        upper_clip: numeric, upper bound to clip data

    Returns:
        Altered array
    """
    x = np.asarray(x)
    return (x + c).clip(lower_clip, upper_clip, out=out)


def downsample(x, n, axis=0):
    """Downsample one dimension of the data.

    The shape of the data is preserved by repeating the values
    in the downsampled axis `n` times.

    Args:
        x: array-like, array to downsample
        n: int, keep one slice every `n` slices
        axis: int, axis in which to downsample

    Returns:
        Downsampled array
    """
    x = np.asarray(x)
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, None, n)
    return np.repeat(x[slices], n, axis=axis)


def flip(x, axis=0):
    """Reverse the order of elements along the given axis.

    Args:
        x: array-like, array to flip
        axis: int, axis on which to flip

    Returns:
        Array with values reversed in one axis
    """
    x = np.asarray(x)
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, None, -1)
    return x[slices]


def from_blocks(a, output_shape):
    """Combine 4D array of non-overlapping blocks `a` into 3D array of shape
    `output_shape`.

    For the reverse of this function, see `to_blocks`.

    Args:
        a: array-like, 4D array of blocks with shape (N, *block_shape), where
            N is the number of blocks.
        output_shape: tuple of len 3, shape of the combined array.
    """
    a = np.asarray(a)

    if a.ndim != 4:
        raise ValueError("This function only works for 4D arrays.")
    if len(output_shape) != 3:
        raise ValueError("output_shape must have three values.")

    n_blocks = a.shape[0]
    block_shape = a.shape[1:]
    ncbrt = np.cbrt(n_blocks).round(6)
    if not ncbrt.is_integer():
        raise ValueError("Cubed root of number of blocks is not an integer")
    ncbrt = int(ncbrt)
    intershape = (ncbrt, ncbrt, ncbrt, *block_shape)

    return (
        a.reshape(intershape)
        .transpose((0, 3, 1, 4, 2, 5))
        .reshape(output_shape))


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
                if arr[ixs, iys, izs].shape != tuple(kernel_size):
                    raise ValueError(
                        "block should have shape {} but got {}".format(
                            kernel_size, arr[ixs, iys, izs].shape))
                yield arr[ixs, iys, izs]


def itervolumes(filepaths,
                block_shape,
                x_dtype,
                y_dtype,
                strides=(1, 1, 1),
                shuffle=False,
                normalizer=None):
    """Yield tuples of numpy arrays `(features, labels)` from a list of
    filepaths to neuroimaging files.
    """
    filepaths = copy.deepcopy(filepaths)

    if shuffle:
        random.shuffle(filepaths)

    for idx, (features_fp, labels_fp) in enumerate(filepaths):
        try:
            features = read_volume(features_fp, dtype=x_dtype)
            labels = read_volume(labels_fp, dtype=y_dtype)
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


def match_histogram(x, target, bins=255):
    """Match the histogram of the `target` array.

    Args:
        x: array-like, array to modify
        target: array-like, array to match
        bins: number of bins to compute in histogram

    Returns:
        Modified array
    """
    x = np.asarray(x)
    target = np.asarray(target)
    sh, _ = np.histogram(x.flatten(), bins=bins, density=True)
    th, tbins = np.histogram(target.flatten(), bins=bins, density=True)
    scdf = sh.cumsum()
    tcdf = th.cumsum()
    return np.interp(
        np.interp(
            x.flatten(),
            tbins[:-1],
            scdf),
        tcdf,
        tbins[:-1]).reshape(x.shape)


def normalize_zero_one(a):
    """Return array with values of `a` normalized to range [0, 1].

    This procedure is also known as min-max scaling.
    """
    a = np.asarray(a)
    min_ = a.min()
    return (a - min_) / (a.max() - min_)


def reduce_contrast(x, out=None):
    """Naively reduces contrast by taking square root of all values.

    Args:
        x: array-like, data to modify

    Returns:
        Modified array
    """
    return np.sqrt(x, out=out)


# https://stackoverflow.com/a/47171600
def replace(a, mapping, assume_all_present=False, zero=True):
    """Replace values in array `a` using dictionary `mapping`.

    Args:
        a: ndarray
        mapping: dict, items in `a` matching a key in `mapping` are replaced
            with the corresponding value. Keys and values may overlap.
        assume_all_present: boolean, true if there is key for each unique value
            in `a`. This allows the use of a faster implementation.
        zero: boolean, zero values in `a` not in `mapping.keys()`

    Returns:
        replaced ndarray
    """
    # Extract keys and values
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))
    sidx = k.argsort()
    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks, a)

    if not assume_all_present:
        idx[idx == len(vs)] = 0
        mask = ks[idx] == a
        out = np.where(mask, vs[idx], a)
    else:
        out = vs[idx]

    if zero:
        out[~np.isin(out, list(mapping.values()))] = 0

    return out


def rotate(x, angle, axes=(1, 0), out=None):
    """Rotate volume.

    Remember to rotate labels if rotating features.

    Args:
        x: array-like, array to rotate
        angle: numeric, angle of rotation in degrees
        axes: tuple of 2 ints, axes that define plane of rotation

    Returns:
        Rotated array.
    """
    x = np.asarray(x)
    if [ax for ax in axes if ax >= x.ndim]:
        raise ValueError("invalid axes for input with {} dims".format(x.ndim))
    return ndimage.rotate(
        input=x,
        angle=angle,
        axes=axes,
        reshape=False,
        output=out)


def salt_and_pepper(x, sval=255, pval=0, stdev_threshold=2, copy=False):
    """Add salt and pepper noise.

    Args:
        x: array-like, array to modify
        sval: salt value
        pval: pepper value
        stdev_threshold: numeric

    Returns:
        Array with salt and pepper noise
    """
    x = np.asarray(x)
    if copy:
        x = x.copy()
    rand = np.random.normal(size=x.shape)
    x[rand < -stdev_threshold] = pval
    x[rand > stdev_threshold] = sval
    return x


def shift(x, s, out=None):
    """Shift array.

    Args:
        x: array-like, array to shift
        s: The shift along the axes. If a float, `shift` is the same
            for each axis. If a sequence, `shift` should contain one value
            for each axis.

    Returns:
        Shifted array
    """
    x = np.asarray(x)
    return ndimage.shift(input=x, shift=s, output=out)


def to_blocks(a, block_shape):
    """Return new array of non-overlapping blocks of shape `block_shape` from
    array `a`.

    For the reverse of this function (blocks to array), see `from_blocks`.

    Args:
        a: array-like, 3D array to block
        block_shape: tuple of len 3, shape of non-overlapping blocks
    """
    a = np.asarray(a)
    orig_shape = np.asarray(a.shape)

    if a.ndim != 3:
        raise ValueError("This function only supports 3D arrays.")
    if len(block_shape) != 3:
        raise ValueError("block_shape must have three values.")

    blocks = orig_shape // block_shape
    inter_shape = tuple(e for tup in zip(blocks, block_shape) for e in tup)
    new_shape = (-1,) + block_shape
    perm = (0, 2, 4, 1, 3, 5)
    return a.reshape(inter_shape).transpose(perm).reshape(new_shape)


# https://stackoverflow.com/a/37121993/5666087
def zoom(x, zoom_factor, **kwds):
    """Zoom and retain shape of input.

    Args:
        zoom_factor: float, > 1 to zoom in, < 1 to zoom out.
    """
    x = np.asarray(x)
    # Zooming out
    if zoom_factor < 1:
        zz = np.round(np.array(x.shape) * zoom_factor).astype(int)
        i, j, k = (x.shape - zz) // 2
        zi, zj, zk = zz
        out = np.zeros_like(x)
        out[i:i + zi, j:j + zj, k:k + zk] = ndimage.zoom(
            x, zoom_factor, **kwds)
    # Zooming in
    elif zoom_factor > 1:
        zz = np.round(np.array(x.shape) / zoom_factor).astype(int)
        i, j, k = (x.shape - zz) // 2
        zi, zj, zk = zz

        out = ndimage.zoom(
            x[i:i + zi, j:j + zj, k:k + zk], zoom_factor, **kwds)

        # Crop if shapes not equal (due to rounding)
        ci, cj, ck = np.array(out.shape) // 2 - (np.array(x.shape) // 2)
        out = out[
            ci:ci + x.shape[0], cj:cj + x.shape[1], ck:ck + x.shape[2]]
    # No zoom
    else:
        out = x

    return out


def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    return (a - a.mean()) / a.std()


class VolumeDataGenerator:
    """Apply real-time data augmentation.

    This object is based on `keras.preprocessing.image.ImageDataGenerator` and
    supports volumetric data instead of images. This object also applies
    relevant augmentations (e.g., rotations, )

    Args:
        samplewise_minmax: boolean, normalize each sample to range [0, 1].
        samplewise_zscore: boolean, zscore each sample.
        samplewise_center: boolean, subtract mean from each sample.
        samplewise_std_normalization: boolean, divide each sample by its
            standard deviation.
        flip: boolean, reverse values in a random axis. If provided, labels are
            flipped in the same way.
        rescale: float, multiply data by this value.
        rotate: boolean, rotate 90 degrees in a random plane. If provided,
            labels are rotated in the same way.
        gaussian: boolean, add Gaussian noise.
        reduce_contrast: boolean, reduce contrast by taking square root of
            (data + 1). One is added to the data before calculating square root
            to avoid amplifying values in range (0, 1).
        salt_and_pepper: boolean, add random salt and pepper noise.
        brightness_range: float, add value in range
            [-brightness_range, brightness_range] to data.
        shift_range: float, range of volume shift in a random axis. Range is
            [-shift_range, shift_range]. If provided, labels are shifted in the
            same way.
        zoom_range: float or tuple len 2, if float, zoom by factor in range
            [1 - zoom_range, 1 + zoom_range]. If tuple, zoom by factor in range
            [zoom_range[0], zoom_range[1]]. If provided, labels are zoomed in
            the same way.
        preprocessing_function: callable, modify data using this function
            prior to standardizing. Must accept a 3D array and return a 3D
            array.
        binarize_y: boolean, binarize labels.
        mapping_y: dict, replace labels that match a key in `mapping_y` with
            the corresponding value, and zero all labels that are not in
            `mapping_y.keys()`. For example, if `mapping_y` is `{10: 1, 20: 2}`
            and the labels are `[0, 5, 10, 20]`, the labels will become
            `[0, 0, 1, 2]`. Keys and values can overlap in the mapping.

    Examples:

    Example of using `.flow_from_files(filepaths)`:

    ```python
    datagen = VolumeDataGenerator(
        samplewise_minmax=True,
        flip=True,
        rotate=True,
        gaussian=True,
        salt_and_pepper=True)

    filepaths = [
        ("path/to/0/T1.nii.gz", "path/to/0/mask.nii.gz"),
        ("path/to/1/T1.nii.gz", "path/to/1/mask.nii.gz"),
        ("path/to/2/T1.nii.gz", "path/to/2/mask.nii.gz")]

    generator = datagen.flow_from_files(
        filepaths=filepaths,
        block_shape=(128, 128, 128),
        strides=(128, 128, 128),
        x_dtype=np.float32,
        y_dtype=np.int32,
        shuffle=False)

    for this_x, this_y in generator:
        model.train(x=this_x, y=this_y)
    ```
    """

    def __init__(self,
                 samplewise_minmax=False,
                 samplewise_zscore=False,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 flip=False,
                 rescale=None,
                 rotate=False,
                 gaussian=False,
                 reduce_contrast=False,
                 salt_and_pepper=False,
                 brightness_range=0.,
                 shift_range=0,
                 zoom_range=0.,
                 preprocessing_function=None,
                 binarize_y=False,
                 mapping_y=None):
        self.samplewise_minmax = samplewise_minmax
        self.samplewise_zscore = samplewise_zscore
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.flip = flip
        self.rescale = rescale
        self.rotate = rotate
        self.gaussian = gaussian
        self.reduce_contrast = reduce_contrast
        self.salt_and_pepper = salt_and_pepper
        self.brightness_range = brightness_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.preprocessing_function = preprocessing_function
        self.binarize_y = binarize_y
        self.mapping_y = mapping_y

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError(
                "`zoom_range` should be a float or a tuple or list of two"
                " floats. Got {}".format(zoom_range))

        if binarize_y and mapping_y:
            raise ValueError(
                "`binarize_y` and `mapping_y` cannot be used together.")

    def standardize(self, x, y=None):
        """Apply the normalization configuration to one volume.

        Only normalizes features.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_minmax:
            x = normalize_zero_one(x)
        if self.samplewise_zscore:
            x = zscore(x)
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + 1e-07)

        if self.binarize_y:
            y = binarize(y)
        if self.mapping_y:
            y = replace(
                y, mapping=self.mapping_y, assume_all_present=False, zero=True)

        if y is None:
            return x
        else:
            return (x, y)

    def random_transform(self, x, y=None, seed=None, copy=True):
        """Randomly augment a volume and, where applicable, the corresponding
        labels.

        Args:
            x: 3D array, single volume features
            y: 3D array, single volume labels
            seed: random seed
            copy: if true, copy x and y arrays

        Returns:
            If `y` is `None`, returns array of transformed x. Else, returns
            tuple of transformed `(x, y)`.
        """
        if seed is not None:
            np.random.seed(seed)

        x = np.array(x, copy=copy, dtype=float)
        if y is not None:
            y = np.array(y, copy=copy, dtype=int)

        if self.flip:
            if np.random.random() < 0.5:
                axis = np.random.choice((0, 1, 2))
                x = flip(x, axis=axis)
                if y is not None:
                    y = flip(y, axis=axis)

        if self.rotate:
            if np.random.random() < 0.5:
                a1, a2 = np.random.choice((0, 1, 2), size=2, replace=False)
                x = x.swapaxes(a1, a2)
                if y is not None:
                    y = y.swapaxes(a1, a2)

        if self.shift_range:
            if np.random.random() < 0.5:
                ts = np.random.uniform(-self.shift_range, self.shift_range)
                s = [ts, 0, 0]
                np.random.shuffle(s)
                x = shift(x, s=s)
                if y is not None:
                    y = shift(y, s=s)

        if self.brightness_range:
            if np.random.random() < 0.5:
                x += np.random.uniform(
                    -self.brightness_range, self.brightness_range)

        if self.zoom_range:
            if np.random.random() < 0.5:
                tz = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
                x = zoom(x, tz)
                if y is not None:
                    y = zoom(y, tz)

        if self.salt_and_pepper:
            if np.random.random() < 0.5:
                x = salt_and_pepper(
                    x, sval=x.max(), pval=0, stdev_threshold=2.5)

        if self.gaussian:
            if np.random.random() < 0.5:
                x *= np.random.normal(loc=1., scale=0.15, size=x.shape)

        if self.reduce_contrast:
            if np.random.random() < 0.5:
                # Add 1 to shift range to [1, 2] before taking sqrt.
                x = reduce_contrast(x + 1).astype(float) - 1.

        if y is None:
            return x
        else:
            return (x, y)

    def flow_from_filepaths(self,
                            filepaths,
                            block_shape,
                            strides,
                            x_dtype='float32',
                            y_dtype='int32',
                            shuffle=None):
        """Generate tuples of `(features, labels)` from a list of filepaths.

        Relevant augmentation (e.g., rotation) is applied to labels
        in addition to features. Other augmentation, like salt and pepper, is
        only applied to the features.

        Args:
            filepaths: list of tuples, where first item in each tuple is the
                path to the features volume, and the second item is the path
                to the corresponding labels volume.
            block_shape: tuple len 3, shape of output blocks.
            strides: int or tuple len 3, stride in each axis when extracting
                blocks from the original volume. If an int, that stride is used
                on all axes. Use `strides = block_shape` to generate
                non-overlapping blocks.
            x_dtype: str or dtype object, datatype of features output.
            y_dtype: str or dtype object, datatype of labels output.
            shuffle: boolean, shuffle list of filepaths.

        Returns:
            Generator of tuples `(features, labels)`. Features have shape
            `(*block_shape, 1)` and labels have shape `block_shape`.
        """
        if shuffle is None:
            raise ValueError(
                "`shuffle` must be explicitly set to True or False.")

        def normalizer(x, y):
            x, y = self.random_transform(x=x, y=y)
            x, y = self.standardize(x=x, y=y)
            return x.astype(x_dtype), y.astype(y_dtype)

        return itervolumes(
            filepaths=filepaths,
            block_shape=block_shape,
            strides=strides,
            x_dtype=x_dtype,
            y_dtype=y_dtype,
            shuffle=shuffle,
            normalizer=normalizer)

    def dset_input_fn_builder(self,
                              filepaths,
                              block_shape,
                              strides,
                              x_dtype,
                              y_dtype,
                              shuffle=None,
                              batch_size=8,
                              n_epochs=1,
                              prefetch=0,
                              multi_gpu=False):
        """Return function that returns instance of `tensorflow.data.Dataset`.

        Dataset generates tuples of augmented `(features, labels)` from a list
        of filepaths.

        Relevant augmentation (e.g., rotation) is applied to labels
        in addition to features. Other augmentation, like salt and pepper, is
        only applied to the features.

        Args:
            filepaths: list of tuples, where first item in each tuple is the
                path to the features volume, and the second item is the path
                to the corresponding labels volume.
            block_shape: tuple len 3, shape of output blocks.
            strides: int or tuple len 3, stride in each axis when extracting
                blocks from the original volume. If an int, that stride is used
                on all axes. Use `strides = block_shape` to generate
                non-overlapping blocks.
            x_dtype: str or dtype object, datatype of features output.
            y_dtype: str or dtype object, datatype of labels output.
            shuffle: boolean, shuffle list of filepaths.
            batch_size: int, number of blocks per batch.
            n_epochs: int, number of epochs.
            prefetch: int, number of blocks to prefetch. See
                `tensorflow.data.Dataset.prefetch`.
            multi_gpu: boolean, train on multiple GPUs.

        Returns:
            Function that returns an instance of `tf.data.Dataset`.
        """
        if shuffle is None:
            raise ValueError(
                "`shuffle` must be explicitly set to True or False.")

        def generator_builder():
            return self.flow_from_filepaths(
                filepaths=filepaths,
                block_shape=block_shape,
                strides=strides,
                x_dtype=x_dtype,
                y_dtype=y_dtype,
                shuffle=shuffle)

        def input_fn():
            """Input function meant to be used with `tf.estimator.Estimator`.
            """
            dset = tf.data.Dataset.from_generator(
                generator=generator_builder,
                output_types=(tf.as_dtype(x_dtype), tf.as_dtype(y_dtype)),
                output_shapes=((*block_shape, 1), block_shape))

            # Loop through the dataset `n_epochs` times.
            dset = dset.repeat(n_epochs)

            if multi_gpu:
                # If the last batch is smaller than `batch_size`, do not use
                # that last batch. This is necessary when training on multiple
                # GPUs because the batch size must always be divisible by the
                # number of GPUs.
                dset = dset.apply(
                    tf.contrib.data.batch_and_drop_remainder(batch_size))
            else:
                # If not training on multiple GPUs, batch sizes do not have to
                # be consistent.
                dset = dset.batch(batch_size)

            if prefetch:
                # Prefetch samples to load/augment new data while training.
                dset = dset.prefetch(prefetch)

            return dset

        return input_fn


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
                "Invalid combination of input shape {}, kernel size {}, and"
                " strides {}. This combination would create a non-integer or"
                " <1 number of blocks."
                .format(arr_shape, kernel_size, strides))

    return tuple(map(int, n_blocks))
