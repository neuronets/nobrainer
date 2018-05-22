# -*- coding: utf-8 -*-
"""Methods to predict using trained models."""

from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf

from nobrainer.volume import from_blocks
from nobrainer.volume import normalize_zero_one
from nobrainer.volume import to_blocks

DT_X = "float32"
_INFERENCE_CLASSES_KEY = "class_ids"


def predict(inputs,
            predictor,
            block_shape,
            normalizer=normalize_zero_one,
            dtype=DT_X):
    """Return predictions from `inputs`.

    This is a general prediction method that can accept various types of
    `inputs` and `predictor`.
    """
    predictor = _get_predictor(predictor)

    if isinstance(inputs, np.ndarray):
        out = predict_from_array(
            inputs=inputs,
            predictor=predictor,
            block_shape=block_shape,
            normalizer=normalize_zero_one)
    elif isinstance(inputs, nib.spatialimages.SpatialImage):
        out = predict_from_img(
            img=inputs,
            predictor=predictor,
            block_shape=block_shape,
            normalizer=normalizer,
            dtype=dtype)
    elif isinstance(inputs, str):
        out = predict_from_filepath(
            filepath=inputs,
            predictor=predictor,
            block_shape=block_shape,
            normalizer=normalizer,
            dtype=dtype)
    elif isinstance(inputs, (list, tuple)):
        out = predict_from_filepaths(
            filepaths=inputs,
            predictor=predictor,
            block_shape=block_shape,
            normalizer=normalizer,
            dtype=dtype)
    return out


def predict_from_array(inputs,
                       predictor,
                       block_shape,
                       normalizer=normalize_zero_one):
    """Return a prediction given a filepath and an ndarray of features.

    Args:
        inputs: ndarray, array of features.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.

    Returns:
        ndarray of predictions.
    """
    if normalizer:
        features = normalizer(inputs)
    features = to_blocks(features, block_shape=block_shape)
    outputs = np.zeros_like(features)
    features = features[..., None]  # Add a dimension for single channel.

    # Predict per block to reduce memory consumption.
    msg = "++ predicting block {} of {}"
    for j in range(features.shape[0]):
        print(msg.format(j + 1, features.shape[0]))
        outputs[j:j + 1] = predictor(
            {'volume': features[j:j + 1]})[_INFERENCE_CLASSES_KEY]

    return from_blocks(outputs, output_shape=inputs.shape)


def predict_from_img(img,
                     predictor,
                     block_shape,
                     normalizer=normalize_zero_one,
                     dtype=DT_X):
    """Return a prediction given a Nibabel image instance and a predictor.

    Args:
        img: nibabel image, image on which to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        dtype: str or dtype object, dtype of features.

    Returns:
        `nibabel.spatialimages.SpatialImage` of predictions.
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
        normalizer=normalizer)
    return nib.spatialimages.SpatialImage(
        dataobj=y, affine=img.affine, header=img.header, extra=img.extra)


def predict_from_filepath(filepath,
                          predictor,
                          block_shape,
                          normalizer=normalize_zero_one,
                          dtype=DT_X):
    """Return a prediction given a filepath and Predictor object.

    Args:
        filepath: path-like, path to existing neuroimaging volume on which
            to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        dtype: str or dtype object, dtype of features.

    Returns:
        `nibabel.spatialimages.SpatialImage` of predictions.
    """
    if not Path(filepath).is_file():
        raise FileNotFoundError("could not find file {}".format(filepath))
    img = nib.load(filepath)
    return predict_from_img(
        img=img,
        predictor=predictor,
        block_shape=block_shape,
        normalizer=normalizer)


def predict_from_filepaths(filepaths,
                           predictor,
                           block_shape,
                           normalizer=normalize_zero_one,
                           dtype=DT_X):
    """Yield predictions from filepaths using a SavedModel.

    Args:
        filepaths: list, neuroimaging volume filepaths on which to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns
            an ndarray. Called before separating volume into blocks.
        dtype: str or dtype object, dtype of features.

    Returns:
        Generator object that yields a `nibabel.spatialimages.SpatialImage` of
        predictions per filepath in list of input filepaths.
    """
    for filepath in filepaths:
        yield predict_from_filepath(
            filepath=filepath,
            predictor=predictor,
            block_shape=block_shape,
            normalizer=normalizer,
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
