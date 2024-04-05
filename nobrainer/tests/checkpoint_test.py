"""Tests for `nobrainer.processing.checkpoint`."""

import os

import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf

from nobrainer.dataset import Dataset
from nobrainer.models import meshnet
from nobrainer.processing.segmentation import Segmentation


def _get_toy_dataset():
    data_shape = (8, 8, 8, 8, 1)
    train = tf.data.Dataset.from_tensors(
        (np.random.rand(*data_shape), np.random.randint(0, 1, data_shape))
    )
    return Dataset(train, data_shape[0], data_shape[1:4], 1)


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
    model1 = Segmentation.init_with_checkpoints(
        meshnet,
        checkpoint_filepath=checkpoint_filepath,
    )
    model1.fit(
        dataset_train=train,
        epochs=2,
    )

    model2 = Segmentation.init_with_checkpoints(
        meshnet,
        checkpoint_filepath=checkpoint_filepath,
    )
    _assert_model_weights_allclose(model1, model2)
    model2.fit(
        dataset_train=train,
        epochs=3,
    )

    model3 = Segmentation.init_with_checkpoints(
        meshnet,
        checkpoint_filepath=checkpoint_filepath,
    )
    _assert_model_weights_allclose(model2, model3)


def test_warm_start_workflow(tmp_path):
    train = _get_toy_dataset()

    checkpoint_dir = os.path.join(tmp_path, "checkpoints")
    checkpoint_filepath = os.path.join(checkpoint_dir, "{epoch:03d}")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    for iteration in range(2):
        bem = Segmentation.init_with_checkpoints(
            meshnet,
            checkpoint_filepath=checkpoint_filepath,
        )
        if iteration == 0:
            assert bem.model is None
        else:
            assert bem.model is not None
            for layer in bem.model.layers:
                for weight_array in layer.get_weights():
                    assert np.count_nonzero(weight_array)
        bem.fit(
            dataset_train=train,
            epochs=2,
        )
