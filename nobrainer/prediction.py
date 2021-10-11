"""Methods to predict using trained models."""

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf

from .transform import get_affine, warp
from .utils import StreamingStats
from .volume import from_blocks, from_blocks_numpy, standardize_numpy, to_blocks_numpy


def predict(
    inputs,
    model,
    block_shape,
    batch_size=1,
    normalizer=None,
    n_samples=1,
    return_variance=False,
    return_entropy=False,
):
    """Return predictions from `inputs`.

    This is a general prediction method that can accept various types of
    `inputs`.

    Parameters
    ----------
    inputs: 3D array or Nibabel image or filepath or list of filepaths.
    model: str: path to saved model, either HDF5 or SavedModel.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.

    Returns
    -------
    If `inputs` is a:
        - 3D numpy array, return an iterable of maximum 3 elements;
            3D array of mean, variance(optional),and entropy(optional) of prediction.
            if the flags for variance or entropy is set False, it won't be returned at
            all The specific order of the elements are:
            mean, variance(default=None) , entropy(default=None)
            Note, variance is only defined when n_sample  > 1
        - Nibabel image or filepath, return a set of Nibabel images of mean, variance,
            entropy of predictions or just the pure arrays of them,
            if return_array_from_images is True.
        - list of filepaths, return generator that yields one set of Nibabel images
            or arrays(if return_array_from_images is set True) of means, variance, and
            entropy predictions per iteration.
    """
    if n_samples < 1:
        raise Exception("n_samples cannot be lower than 1.")

    model = _get_model(model)

    if isinstance(inputs, np.ndarray):
        out = predict_from_array(
            inputs=inputs,
            model=model,
            block_shape=block_shape,
            batch_size=batch_size,
            normalizer=normalizer,
            n_samples=n_samples,
            return_variance=return_variance,
            return_entropy=return_entropy,
        )
    elif isinstance(inputs, nib.spatialimages.SpatialImage):
        out = predict_from_img(
            img=inputs,
            model=model,
            block_shape=block_shape,
            batch_size=batch_size,
            normalizer=normalizer,
            n_samples=n_samples,
            return_variance=return_variance,
            return_entropy=return_entropy,
        )
    elif isinstance(inputs, str):
        out = predict_from_filepath(
            filepath=inputs,
            model=model,
            block_shape=block_shape,
            batch_size=batch_size,
            normalizer=normalizer,
            n_samples=n_samples,
            return_variance=return_variance,
            return_entropy=return_entropy,
        )
    elif isinstance(inputs, (list, tuple)):
        out = predict_from_filepaths(
            filepaths=inputs,
            model=model,
            block_shape=block_shape,
            batch_size=batch_size,
            normalizer=normalizer,
            n_samples=n_samples,
            return_variance=return_variance,
            return_entropy=return_entropy,
        )
    else:
        raise TypeError("Input to predict is not a valid type")
    return out


def predict_from_array(
    inputs,
    model,
    block_shape,
    batch_size=1,
    normalizer=None,
    n_samples=1,
    return_variance=False,
    return_entropy=False,
):
    """Return a prediction given a filepath and an ndarray of features.

    Parameters
    ----------
    inputs: ndarray, array of features.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.

    Returns
    -------
    ndarray of predictions.
    """
    if normalizer:
        features = normalizer(inputs)
    else:
        features = inputs
    if block_shape is not None:
        features = to_blocks_numpy(features, block_shape=block_shape)
    else:
        features = features[None]  # Add batch dimension.

    # Add a dimension for single channel.
    features = features[..., None]

    # Predict per block to reduce memory consumption.
    n_blocks = features.shape[0]
    n_batches = math.ceil(n_blocks / batch_size)

    if not return_variance and not return_entropy and n_samples == 1:
        # TODO: has better performance but output should change to numpy array
        # outputs = model(features).numpy()
        outputs = model.predict(features, batch_size=1, verbose=0)
        if outputs.shape[-1] == 1:
            # Binarize according to threshold.
            outputs = outputs > 0.3
            outputs = outputs.squeeze(-1)
            # Nibabel doesn't like saving boolean arrays as Nifti.
            outputs = outputs.astype(np.uint8)
        else:
            # Hard classes for multi-class segmentation.
            outputs = np.argmax(outputs, -1)
        outputs = from_blocks_numpy(outputs, output_shape=inputs.shape)
        return outputs
    else:
        # Bayesian prediction based on sampling from distributions
        means = np.zeros_like(features.squeeze(-1))
        variances = np.zeros_like(features.squeeze(-1))
        entropies = np.zeros_like(features.squeeze(-1))
        progbar = tf.keras.utils.Progbar(n_batches)
        progbar.update(0)
        for j in range(0, n_blocks, batch_size):

            this_x = features[j : j + batch_size]
            s = StreamingStats()
            for n in range(n_samples):
                # TODO: has better performance but output should change to numpy array
                # new_prediction = model(this_x).numpy()
                new_prediction = model.predict(this_x, batch_size=1, verbose=0)
                s.update(new_prediction)

            means[j : j + batch_size] = np.argmax(s.mean(), axis=-1)  # max mean
            variances[j : j + batch_size] = np.sum(s.var(), axis=-1)
            entropies[j : j + batch_size] = np.sum(s.entropy(), axis=-1)  # entropy
            progbar.add(1)

        total_means = from_blocks_numpy(means, output_shape=inputs.shape)
        total_variance = from_blocks_numpy(variances, output_shape=inputs.shape)
        total_entropy = from_blocks_numpy(entropies, output_shape=inputs.shape)

    include_variance = (n_samples > 1) and (return_variance)
    if include_variance:
        if return_entropy:
            return total_means, total_variance, total_entropy
        else:
            return total_means, total_variance
    else:
        if return_entropy:
            return total_means, total_entropy
        else:
            return total_means


def predict_from_img(
    img,
    model,
    block_shape,
    batch_size=1,
    normalizer=None,
    n_samples=1,
    return_variance=False,
    return_entropy=False,
):
    """Return a prediction given a Nibabel image instance and a predictor.

    Parameters
    ----------
    img: nibabel image, image on which to predict.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.

    Returns
    -------
    `nibabel.spatialimages.SpatialImage` or arrays of prediction of mean,
    variance(optional) or entropy (optional).
    """
    if not isinstance(img, nib.spatialimages.SpatialImage):
        raise ValueError("image is not a nibabel image type")
    inputs = np.asarray(img.dataobj)
    img.uncache()
    inputs = inputs.astype(np.float32)

    y = predict_from_array(
        inputs=inputs,
        model=model,
        block_shape=block_shape,
        batch_size=batch_size,
        normalizer=normalizer,
        n_samples=n_samples,
        return_variance=return_variance,
        return_entropy=return_entropy,
    )

    if isinstance(y, np.ndarray):
        return nib.spatialimages.SpatialImage(
            dataobj=y, affine=img.affine, header=img.header, extra=img.extra
        )
    else:
        if len(y) == 2:
            return (
                nib.spatialimages.SpatialImage(
                    dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra
                ),
                nib.spatialimages.SpatialImage(
                    dataobj=y[1], affine=img.affine, header=img.header, extra=img.extra
                ),
            )
        elif len(y) == 3:
            return (
                nib.spatialimages.SpatialImage(
                    dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra
                ),
                nib.spatialimages.SpatialImage(
                    dataobj=y[1], affine=img.affine, header=img.header, extra=img.extra
                ),
                nib.spatialimages.SpatialImage(
                    dataobj=y[2], affine=img.affine, header=img.header, extra=img.extra
                ),
            )


def predict_from_filepath(
    filepath,
    model,
    block_shape,
    batch_size=1,
    normalizer=None,
    n_samples=1,
    return_variance=False,
    return_entropy=False,
):
    """Predict on a volume given a filepath and a trained model.

    Parameters
    ----------
    filepath: path-like, path to existing neuroimaging volume on which
        to predict.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.

    Returns
    -------
    `nibabel.spatialimages.SpatialImage` or arrays of predictions of
        mean, variance(optional), and entropy (optional).
    """
    if not Path(filepath).is_file():
        raise FileNotFoundError("could not find file {}".format(filepath))
    img = nib.load(filepath)
    return predict_from_img(
        img=img,
        model=model,
        block_shape=block_shape,
        batch_size=batch_size,
        normalizer=normalizer,
        n_samples=n_samples,
        return_variance=return_variance,
        return_entropy=return_entropy,
    )


def predict_from_filepaths(
    filepaths,
    model,
    block_shape,
    batch_size=1,
    normalizer=None,
    n_samples=1,
    return_variance=False,
    return_entropy=False,
):
    """Yield a model's predictions on a list of filepaths.

    Parameters
    ----------
    filepaths: list, volume filepaths on which to predict.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.

    Returns
    -------
    Generator object that yields a `nibabel.spatialimages.SpatialImage` or
    arrays of predictions per filepath in list of input filepaths.
    """
    for filepath in filepaths:
        yield predict_from_filepath(
            filepath=filepath,
            model=model,
            block_shape=block_shape,
            batch_size=batch_size,
            normalizer=normalizer,
            n_samples=n_samples,
            return_variance=return_variance,
            return_entropy=return_entropy,
        )


def _get_model(path):
    """Return `tf.keras.Model` object from a filepath.

    Parameters
    ----------
    path: str, path to HDF5 or SavedModel file.

    Returns
    -------
    Instance of `tf.keras.Model`.

    Raises
    ------
    `ValueError` if cannot load model.
    """
    if isinstance(path, tf.keras.Model):
        return path
    try:
        return tf.keras.models.load_model(path, compile=False)
    except OSError:
        # Not an HDF5 file.
        pass

    try:
        path = Path(path)
        if path.suffix == ".json":
            path = path.parent.parent
        return tf.keras.experimental.load_from_saved_model(str(path))
    except Exception:
        pass

    raise ValueError(
        "Failed to load model. Is the model in HDF5 format or SavedModel" " format?"
    )


def _get_predictor(model_path):
    """restores the tf estimator model.
    The output is a tf2 estimator"""
    predictor = tf.saved_model.load(model_path)
    return predictor.signatures["serving_default"]


def _transform_and_predict(
    model, x, block_shape, rotation, translation=[0, 0, 0], verbose=False
):
    """Predict on rigidly transformed features.

    The rigid transformation is applied to the volumes prior to prediction, and
    the prediced labels are transformed with the inverse warp, so that they are
    in the same space.

    Parameters
    ----------
    model: `tf.keras.Model`, model used for prediction.
    x: 3D array, volume of features.
    block_shape: tuple of length 3, shape of non-overlapping blocks to take
        from the features. This also corresponds to the input of the model, not
        including the batch or channel dimensions.
    rotation: tuple of length 3, rotation angle in radians in each dimension.
    translation: tuple of length 3, units of translation in each dimension.
    verbose: bool, whether to print progress bar.

    Returns
    -------
    Array of predictions with the same shape and in the same space as the
    original input features.
    """

    x = np.asarray(x).astype(np.float32)
    affine = get_affine(x.shape, rotation=rotation, translation=translation)
    inverse_affine = tf.linalg.inv(affine)
    x_warped = warp(x, affine, order=1)

    x_warped_blocks = to_blocks_numpy(x_warped, block_shape)
    x_warped_blocks = x_warped_blocks[..., np.newaxis]  # add grayscale channel
    x_warped_blocks = standardize_numpy(x_warped_blocks)
    y = model.predict(x_warped_blocks, batch_size=1, verbose=verbose)

    n_classes = y.shape[-1]
    if n_classes == 1:
        y = y.squeeze(-1)
    else:
        # Usually, the argmax would be taken to get the class membership of
        # each voxel, but if we get hard values, then we cannot average
        # multiple predictions.
        raise ValueError(
            "This function is not compatible with multi-class predictions."
        )

    y = from_blocks_numpy(y, x.shape)
    y = warp(y, inverse_affine, order=0).numpy()

    return y


def predict_by_estimator(
    filepath,
    model_path,
    block_shape,
    batch_size=4,
    normalizer=None,
    n_samples=1,
    return_variance=True,
    return_entropy=True,
):
    """Predict on a volume given a filepath and a path to tensorflow1 estimator
    saved model.

    Parameters
    ----------
    filepath: path-like, path to existing neuroimaging volume on which
        to predict.
    model_path: path_like, path to saved tf1 estimator model, trained model.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.

    Returns
    -------
    `nibabel.spatialimages.SpatialImage` or arrays of predictions of
        mean, variance(optional), and entropy (optional).
    """

    if not Path(filepath).is_file():
        raise FileNotFoundError("could not find file {}".format(filepath))

    img = nib.load(filepath)
    inputs = np.asarray(img.dataobj)
    img.uncache()
    inputs = inputs.astype(np.float32)

    if normalizer:
        features = normalizer(inputs)
    else:
        features = inputs

    features = to_blocks_numpy(features, block_shape=block_shape)

    # Add a dimension for single channel.
    features = features[..., None]

    # restores the tf estimator model
    predictor = _get_predictor(model_path)

    # Predict per block to reduce memory consumption.
    n_blocks = features.shape[0]
    n_batches = math.ceil(n_blocks / batch_size)

    # Variational inference
    means = np.zeros_like(features.squeeze(-1))
    variances = np.zeros_like(features.squeeze(-1))
    entropies = np.zeros_like(features.squeeze(-1))
    progbar = tf.keras.utils.Progbar(n_batches)
    progbar.update(0)
    for j in range(0, n_blocks, batch_size):
        this_x = tf.convert_to_tensor(features[j : j + batch_size])
        s = StreamingStats()
        for n in range(n_samples):
            new_prediction = predictor(this_x)
            s.update(new_prediction["probabilities"])

        means[j : j + batch_size] = np.argmax(s.mean(), axis=-1)  # max mean
        variances[j : j + batch_size] = np.sum(s.var(), axis=-1)
        entropies[j : j + batch_size] = np.sum(s.entropy(), axis=-1)  # entropy
        progbar.add(1)

        total_means = from_blocks(means, output_shape=inputs.shape).numpy()
        total_variance = from_blocks(variances, output_shape=inputs.shape).numpy()
        total_entropy = from_blocks(entropies, output_shape=inputs.shape).numpy()

    include_variance = (n_samples > 1) and (return_variance)
    if include_variance:
        if return_entropy:
            # return in the form of a nifti image
            total_means = nib.spatialimages.SpatialImage(
                dataobj=total_means,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )

            total_variance = nib.spatialimages.SpatialImage(
                dataobj=total_variance,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )

            total_entropy = nib.spatialimages.SpatialImage(
                dataobj=total_entropy,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )
            return total_means, total_variance, total_entropy
        else:
            total_means = nib.spatialimages.SpatialImage(
                dataobj=total_means,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )

            total_variance = nib.spatialimages.SpatialImage(
                dataobj=total_variance,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )
            return total_means, total_variance
    else:
        if return_entropy:
            total_means = nib.spatialimages.SpatialImage(
                dataobj=total_means,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )

            total_entropy = nib.spatialimages.SpatialImage(
                dataobj=total_entropy,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )
            return total_means, total_entropy
        else:
            total_means = nib.spatialimages.SpatialImage(
                dataobj=total_means,
                affine=img.affine,
                header=img.header,
                extra=img.extra,
            )
            return total_means
