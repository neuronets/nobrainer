# -*- coding: utf-8 -*-
"""Methods to predict using trained models."""

import tensorflow as tf

from nobrainer.io import read_volume
from nobrainer.volume import as_blocks
from nobrainer.volume import from_blocks
from nobrainer.volume import normalize_zero_one


def predict(savedmodel_dir,
            filepaths,
            block_shape,
            normalizer=normalize_zero_one,
            dtype="float32"):
    """Yield predictions from filepaths using a SavedModel.

    Args:
        savedmodel_dir: path-like, path to SavedModel directory.
        filepaths: list, neuroimaging volume filepaths on which to predict.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns
            an ndarray. Called before separating volume into blocks.
        dtype: str or dtype object, dtype of features.

    Returns:
        Generator object.
    """
    predictor = tf.contrib.predictor.from_saved_model(savedmodel_dir)
    for filepath in filepaths:
        yield predict_filepath(
            predictor=predictor,
            filepath=filepath,
            block_shape=block_shape,
            normalizer=normalizer,
            dtype=dtype)


def predict_filepath(predictor,
                     filepath,
                     block_shape,
                     normalizer=normalize_zero_one,
                     dtype="float32"):
    """Return a prediction given a filepath and Predictor object.

    Args:
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        filepath: path-like, path to existing neuroimaging volume on which
            to predict.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        dtype: str or dtype object, dtype of features.

    Returns:
        Ndarray of predictions.
    """
    inputs = read_volume(filepath, dtype=dtype)
    return predict_array(
        predictor=predictor,
        inputs=inputs,
        block_shape=block_shape,
        normalizer=normalizer)


def predict_array(predictor,
                  inputs,
                  block_shape,
                  normalizer=normalize_zero_one):
    """Return a prediction given a filepath and an ndarray of features.

    Args:
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        inputs: ndarray, array of features.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.

    Returns:
        Ndarray of predictions.
    """
    if normalizer:
        features = normalizer(inputs)
    features = as_blocks(features, block_shape=block_shape)
    features = features[..., None]  # Add a dimension for single channel.
    predictions = predictor({'volume': features})
    return from_blocks(predictions, output_shape=inputs.shape)
