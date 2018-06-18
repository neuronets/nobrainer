# -*- coding: utf-8 -*-
"""Methods to predict using trained models."""

import math
from pathlib import Path
import nibabel as nib
import numpy as np
import tensorflow as tf
import time
from nobrainer.volume import from_blocks
from nobrainer.volume import normalize_zero_one
from nobrainer.volume import to_blocks

DT_X = "float32"
_INFERENCE_CLASSES_KEY = "class_ids"


def predict(inputs,
            predictor,
            block_shape,
            returnVariance=False,
            returnEntropy=False,
            returnArrayFromImages = False, 
            n_samples=1,
            normalizer=normalize_zero_one,
            batch_size=4,
            dtype=DT_X):
    """Return predictions from `inputs`.

    This is a general prediction method that can accept various types of
    `inputs` and `predictor`.

    Args:
        inputs: 3D array or Nibabel image or filepath or list of filepaths.
        predictor: path-like or TensorFlow Predictor object, if path, must be
            path to SavedModel.
        returnVariance: Boolean. If set True, it returns the running population 
            variance along with mean. Note, if the n_samples is smaller or equal to 1,
            the variance will not be returned; instead it will return None
        returnEntropy: Boolean. If set True, it returns the running entropy.
            along with mean.       
        returnArrayFromImages: Boolean. If set True and the given input is either image,
            filepath, or filepaths, it will return arrays of [mean, variance, entropy]
            instead of images of them. Also, if the input is array, it will
            simply return array, whether or not this flag is True or False.
        n_samples: The number of sampling. If set as 1, it will just return the 
            single prediction value. The default value is 1
        block_shape: tuple of length 3, shape of sub-volumes on which to
            predict.
        normalizer: callable, function that accepts two arguments
            `(features, labels)` and returns a tuple of modified
            `(features, labels)`.
        batch_size: int, number of sub-volumes per batch for predictions.
        dtype: str or dtype object, datatype of features array.

    Returns:
        If `inputs` is a:
            -3D numpy array, return an iterable of maximum 3 elements;
                3D array of mean, variance(optional),and entropy(optional) of prediction. 
                if the flags for variance or entropy is set False, it won't be returned at all
                The specific order of the elements are:
                mean, variance(default=None) , entropy(default=None)
                Note, variance is only defined when n_sample  > 1
            - Nibabel image or filepath, return a set of Nibabel images of mean, variance, 
                entropy of predictions or just the pure arrays of them, 
                if returnArrayFromImages is True.
            - list of filepaths, return generator that yields one set of Nibabel images
                or arrays(if returnArrayFromImages is set True) of means, variance, and
                entropy predictions per iteration.
    """
    predictor = _get_predictor(predictor)

    if isinstance(inputs, np.ndarray):
        out = predict_from_array(
            inputs=inputs,
            predictor=predictor,
            block_shape=block_shape,
            returnVariance=returnVariance,
            returnEntropy=returnEntropy,
            returnArrayFromImages=returnArrayFromImages, 
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size)
    elif isinstance(inputs, nib.spatialimages.SpatialImage):
        out = predict_from_img(
            img=inputs,
            predictor=predictor,
            block_shape=block_shape,
            returnVariance=returnVariance,
            returnEntropy=returnEntropy,
            returnArrayFromImages=returnArrayFromImages, 
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)
    elif isinstance(inputs, str):
        out = predict_from_filepath(
            filepath=inputs,
            predictor=predictor,
            block_shape=block_shape,
            returnVariance=returnVariance,
            returnEntropy=returnEntropy,
            returnArrayFromImages=returnArrayFromImages, 
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)
    elif isinstance(inputs, (list, tuple)):
        out = predict_from_filepaths(
            filepaths=inputs,
            predictor=predictor,
            block_shape=block_shape,
            nreturnVariance=returnVariance,
            returnEntropy=returnEntropy,
            returnArrayFromImages=returnArrayFromImages, 
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)
    return out


def predict_from_array(inputs,
                       predictor,
                       block_shape,
                       returnVariance=False,
                       returnEntropy=False,
                       returnArrayFromImages=False, 
                       n_samples=1,
                       normalizer=normalize_zero_one,
                       batch_size=4):
    """Return a prediction given a filepath and an ndarray of features.

    Args:
        inputs: ndarray, array of features.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        returnVariance: 'y' or 'n'. If set True, it returns the running population
            variance along with mean. Note, if the n_samples is smaller or equal to 1,
            the variance will not be returned; instead it will return None
        returnEntropy: Boolean. If set True, it returns the running entropy.
            along with mean.       
        returnArrayFromImages: Boolean. If set True and the given input is either image,
            filepath, or filepaths, it will return arrays of [mean, variance, entropy]
            instead of images of them. Also, if the input is array, it will
            simply return array, whether or not this flag is True or False.
        n_samples: The number of sampling. If set as 1, it will just return the 
            single prediction value.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        batch_size: int, number of sub-volumes per batch for prediction.

    Returns:
        ndarray of predictions.
    """

    if normalizer:
        features = normalizer(inputs)
    else:
        features = inputs
    features = to_blocks(features, block_shape=block_shape) #NAO# This may be the clue.# This divides the image into cubes!
    means = np.zeros_like(features)
    variances = np.zeros_like(features)
    entropies = np.zeros_like(features)

    features = features[..., None]  # Add a dimension for single channel.

    # Predict per block to reduce memory consumption.
    n_blocks = features.shape[0]
    n_batches = math.ceil(n_blocks / batch_size)
    progbar = tf.keras.utils.Progbar(n_batches)
    progbar.update(0)
    for j in range(0, n_blocks, batch_size):

        newPrediction = predictor( {'volume': features[j:j + batch_size]})
        prevMean = np.zeros_like(newPrediction['probabilities'])
        currMean = newPrediction['probabilities']
        
        M = np.zeros_like(newPrediction['probabilities'])
        for n in range(1, n_samples):

            newPrediction = predictor( {'volume': features[j:j + batch_size]})
            prevMean = currMean
            currMean = prevMean + (newPrediction['probabilities'] - prevMean)/float(n+1)
            M = M + np.multiply(prevMean - newPrediction['probabilities'], currMean - newPrediction['probabilities'])

        progbar.add(1)
        means[j:j + batch_size] = np.argmax(currMean, axis = -1 ) # max mean
        variances[j:j + batch_size] = np.sum(M/n_samples, axis = -1)
        entropies[j:j + batch_size] = -np.sum(np.multiply(np.log(currMean+0.001),currMean), axis = -1) # entropy
    totalMeans =from_blocks(means, output_shape=inputs.shape)
    totalVariance = from_blocks(variances, output_shape=inputs.shape)
    totalEntropy = from_blocks(entropies, output_shape=inputs.shape)

    includeVariance = ((n_samples > 1) and (returnVariance))
    if includeVariance:
        if returnEntropy:
            return totalMeans, totalVariance, totalEntropy
        else:
            return totalMeans,totalVariance
    else:
        if returnEntropy:
            return totalMeans, totalEntropy
        else:
            return totalMeans,


def predict_from_img(img,
                     predictor,
                     block_shape,
                     returnVariance=False,
                     returnEntropy=False,
                     returnArrayFromImages=False, 
                     n_samples=1,
                     normalizer=normalize_zero_one,
                     batch_size=4,
                     dtype=DT_X):
    """Return a prediction given a Nibabel image instance and a predictor.

    Args:
        img: nibabel image, image on which to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        returnVariance: Boolean. If set True, it returns the running population 
            variance along with mean. Note, if the n_samples is smaller or equal to 1,
            the variance will not be returned; instead it will return None
        returnEntropy: Boolean. If set True, it returns the running entropy.
            along with mean.       
        returnArrayFromImages: Boolean. If set True and the given input is either image,
            filepath, or filepaths, it will return arrays of [mean, variance, entropy]
            instead of images of them. Also, if the input is array, it will
            simply return array, whether or not this flag is True or False.
        n_samples: The number of sampling. If set as 1, it will just return the 
            single prediction value.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        batch_size: int, number of sub-volumes per batch for prediction.
        dtype: str or dtype object, dtype of features.

    Returns:
        `nibabel.spatialimages.SpatialImage` or arrays of prediction of mean, 
            variance(optional) or entropy (optional).
    """
    if not isinstance(img, nib.spatialimages.SpatialImage):
        raise ValueError("image is not a nibabel image type")
    inputs = np.asarray(img.dataobj)
    if dtype is not None:
        inputs = inputs.astype(dtype)
    img.uncache()
    y = predict_from_array(
        inputs=inputs,
        predictor=predictor,
        block_shape=block_shape,
        returnVariance=returnVariance,
        returnEntropy=returnEntropy,
        returnArrayFromImages=returnArrayFromImages, 
        n_samples=n_samples,
        normalizer=normalizer,
        batch_size=batch_size)

    if returnArrayFromImages:
        return y
    else:
        if len(y) == 1:
            return nib.spatialimages.SpatialImage(
                dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra),

        elif len(y) == 2:
            return nib.spatialimages.SpatialImage(
                dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=y[1], affine=img.affine, header=img.header, extra=img.extra)
        else:           # 3 inputs
            return nib.spatialimages.SpatialImage(
                dataobj=y[0], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=y[1], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=y[2], affine=img.affine, header=img.header, extra=img.extra)





def predict_from_filepath(filepath,
                          predictor,
                          block_shape,
                          returnVariance=False,
                          returnEntropy=False,
                          returnArrayFromImages=False, 
                          n_samples=1,
                          normalizer=normalize_zero_one,
                          batch_size=4,
                          dtype=DT_X):
    """Return a prediction given a filepath and Predictor object.

    Args:
        filepath: path-like, path to existing neuroimaging volume on which
            to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        returnVariance: Boolean. If set True, it returns the running population 
            variance along with mean. Note, if the n_samples is smaller or equal to 1,
            the variance will not be returned; instead it will return None
        returnEntropy: Boolean. If set True, it returns the running entropy.
            along with mean.       
        returnArrayFromImages: Boolean. If set True and the given input is either image,
            filepath, or filepaths, it will return arrays of [mean, variance, entropy]
            instead of images of them. Also, if the input is array, it will
            simply return array, whether or not this flag is True or False.
        n_samples: The number of sampling. If set as 1, it will just return the 
            single prediction value.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        batch_size: int, number of sub-volumes per batch for prediction.
        dtype: str or dtype object, dtype of features.

    Returns:
        `nibabel.spatialimages.SpatialImage` or arrays of predictions of 
        mean, variance(optional), and entropy (optional).
    """
    if not Path(filepath).is_file():
        raise FileNotFoundError("could not find file {}".format(filepath))
    img = nib.load(filepath)
    return predict_from_img(
        img=img,
        predictor=predictor,
        block_shape=block_shape,
        returnVariance=returnVariance,
        returnEntropy=returnEntropy,
        returnArrayFromImages=returnArrayFromImages, 
        n_samples=n_samples,
        normalizer=normalizer,
        batch_size=batch_size)


def predict_from_filepaths(filepaths,
                           predictor,
                           block_shape,
                           returnVariance=False,
                           returnEntropy=False,
                           returnArrayFromImages=False, 
                           n_samples=1,
                           normalizer=normalize_zero_one,
                           batch_size=4,
                           dtype=DT_X):
    """Yield predictions from filepaths using a SavedModel.

    Args:
        filepaths: list, neuroimaging volume filepaths on which to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns
            an ndarray. Called before separating volume into blocks.
        batch_size: int, number of sub-volumes per batch for prediction.
        dtype: str or dtype object, dtype of features.

    Returns:
        Generator object that yields a `nibabel.spatialimages.SpatialImage` or
        arrays of predictions per filepath in list of input filepaths.
    """
    for filepath in filepaths:
        yield predict_from_filepath(
            filepath=filepath,
            predictor=predictor,
            block_shape=block_shape,
            returnVariance=returnVariance,
            returnEntropy=returnEntropy,
            returnArrayFromImages=returnArrayFromImages, 
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype)


def _get_predictor(predictor):
    """Return `tf.contrib.predictor.predictor.Predictor` object from a filepath
    or a `Predictor` object.
    """
    from tensorflow.contrib.predictor.predictor import Predictor

    if isinstance(predictor, Predictor):
        pass
    else:
        try:
            path = Path(predictor)
            # User might provide path to saved_model.pb but predictor expects
            # parent directory.
            if path.suffix == '.pb':
                path = path.parent
            predictor = tf.contrib.predictor.from_saved_model(str(path))
        except Exception:
            raise ValueError(
                "Failed to load predictor. Is `predictor` a path to a saved"
                " model?")
    return predictor




