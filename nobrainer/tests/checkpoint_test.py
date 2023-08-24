"""Tests for `nobrainer.processing.checkpoint`."""

import os

import numpy as np
from numpy.testing import assert_allclose
import pytest
import tensorflow as tf

from nobrainer.models import meshnet
from nobrainer.processing.segmentation import Segmentation


def _get_toy_dataset():
    data_shape = (8, 8, 8, 8, 1)
    train = tf.data.Dataset.from_tensors(
        (np.random.rand(*data_shape), np.random.randint(0, 1, data_shape))
    )
    train.scalar_labels = False
    train.n_volumes = data_shape[0]
    train.volume_shape = data_shape[1:4]
    return train


def _assert_model_weights_allclose(model1, model2):
    for layer1, layer2 in zip(model1.model.layers, model2.model.layers):
        weights1 = layer1.get_weights()
        weights2 = layer2.get_weights()
        assert len(weights1) == len(weights2)
        for index in range(len(weights1)):
            assert_allclose(weights1[index], weights2[index], rtol=1e-06, atol=1e-08)


def test_checkpoint(tmp_path):
    train = _get_toy_dataset()

    checkpoint_filepath = os.path.join(tmp_path, "checkpoint-epoch_{epoch:03d}")
    model1 = Segmentation(meshnet, checkpoint_filepath=checkpoint_filepath)
    model1.fit(
        dataset_train=train,
        epochs=2,
    )

    model2 = Segmentation.load_latest(checkpoint_filepath=checkpoint_filepath)
    _assert_model_weights_allclose(model1, model2)
    model2.fit(
        dataset_train=train,
        epochs=3,
    )

    model3 = Segmentation.load_latest(checkpoint_filepath=checkpoint_filepath)
    _assert_model_weights_allclose(model2, model3)


def test_warm_start_workflow(tmp_path):
    train = _get_toy_dataset()

    checkpoint_dir = os.path.join("checkpoints")
    checkpoint_filepath = os.path.join(checkpoint_dir, "{epoch:03d}")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    try:
        bem = Segmentation.load_latest(checkpoint_filepath=checkpoint_filepath)
    except (AssertionError, ValueError):
        bem = Segmentation(meshnet, checkpoint_filepath=checkpoint_filepath)
    bem.fit(
        dataset_train=train,
        epochs=2,
    )
