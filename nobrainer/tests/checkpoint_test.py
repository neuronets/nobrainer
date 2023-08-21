"""Tests for `nobrainer.processing.checkpoint`."""

import os

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from nobrainer.models import meshnet
from nobrainer.processing.checkpoint import CheckpointTracker
from nobrainer.processing.segmentation import Segmentation


def test_checkpoint(tmp_path):
    data_shape = (8, 8, 8, 8, 1)
    train = tf.data.Dataset.from_tensors(
        (np.random.rand(*data_shape), np.random.randint(0, 1, data_shape))
    )
    train.scalar_labels = False
    train.n_volumes = data_shape[0]
    train.volume_shape = data_shape[1:4]

    checkpoint_file_path = os.path.join(tmp_path, "checkpoint-epoch_{epoch:03d}")
    model1 = Segmentation(meshnet)
    model1.fit(
        dataset_train=train,
        checkpoint_file_path=checkpoint_file_path,
        epochs=2,
    )

    model2 = Segmentation(meshnet)
    checkpoint_tracker = CheckpointTracker(model2, checkpoint_file_path)
    model2 = checkpoint_tracker.load()

    for layer1, layer2 in zip(model1.model.layers, model2.model.layers):
        weights1 = layer1.get_weights()
        weights2 = layer2.get_weights()
        assert len(weights1) == len(weights2)
        for index in range(len(weights1)):
            assert_array_equal(weights1[index], weights2[index])
