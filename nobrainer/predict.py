# -*- coding: utf-8 -*-
"""Methods to predict using trained models."""

# import tensorflow as tf

from nobrainer.io import read_volume
from nobrainer.volume import as_blocks
from nobrainer.volume import from_blocks
from nobrainer.volume import normalize_zero_one


def predict(model_dir, filepaths, block_shape, normalizer=normalize_zero_one):
    """"""
    pass


def predict_array(predictor,
                  inputs,
                  block_shape,
                  normalizer=normalize_zero_one):
    """"""
    features = normalizer(inputs)
    features = as_blocks(features, block_shape=block_shape)
    features = features[..., None]  # Add a dimension for single channel.
    predictions = predictor({'volume': features})
    return from_blocks(predictions, output_shape=inputs.shape)


def predict_filepath(predictor,
                     filepath,
                     block_shape,
                     normalizer=normalize_zero_one,
                     dtype="float32"):
    """"""
    inputs = read_volume(filepath, dtype=dtype)
    return predict_array(
        predictor=predictor,
        inputs=inputs,
        block_shape=block_shape,
        normalizer=normalizer)
