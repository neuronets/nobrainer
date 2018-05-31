# -*- coding: utf-8 -*-
"""Tests for MeshNet."""

import numpy as np
import tensorflow as tf

from nobrainer.models.meshnet import MeshNet


def test_meshnet():
    shape = (1, 5, 5, 5)
    X = np.random.rand(*shape, 1).astype(np.float32)
    y = np.random.randint(0, 9, size=(shape), dtype=np.int32)

    def dset_fn():
        return tf.data.Dataset.from_tensors((X, y))

    estimator = MeshNet(
        n_classes=10,
        optimizer='Adam',
        n_filters=71,
        dropout_rate=0.25,
        learning_rate=0.001)
    estimator.train(input_fn=dset_fn)

    # With optimizer object.
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    estimator = MeshNet(
        n_classes=10,
        optimizer=optimizer,
        n_filters=71,
        dropout_rate=0.25,
        learning_rate=0.001)
    estimator.train(input_fn=dset_fn)
