# -*- coding: utf-8 -*-
"""Tests for HighRes3DNet."""

import numpy as np
import tensorflow as tf

from nobrainer.models.highres3dnet import HighRes3DNet


def test_highres3dnet():
    shape = (1, 5, 5, 5)
    X = np.random.rand(*shape, 1).astype(np.float32)
    y = np.random.randint(0, 9, size=(shape), dtype=np.int32)

    def dset_fn():
        return tf.data.Dataset.from_tensors((X, y))

    estimator = HighRes3DNet(
        n_classes=10,
        optimizer='Adam',
        learning_rate=0.001)
    estimator.train(input_fn=dset_fn)

    # With optimizer object.
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    estimator = HighRes3DNet(
        n_classes=10,
        optimizer=optimizer,
        learning_rate=0.001)
    estimator.train(input_fn=dset_fn)

    # With one batchnorm layer per residually connected pair and dropout.
    estimator = HighRes3DNet(
        n_classes=10,
        optimizer='Adam',
        learning_rate=0.001,
        one_batchnorm_per_resblock=True,
        dropout_rate=0.25)
    estimator.train(input_fn=dset_fn)
