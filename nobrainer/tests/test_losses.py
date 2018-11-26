# -*- coding: utf-8 -*-
"""Tests for `nobrainer.losses`."""

import numpy as np
import tensorflow as tf

from nobrainer import losses


def test_dice():
    with tf.Session() as sess:
        shape = (2, 4, 4)
        dtype = np.float32
        labels = np.zeros(shape=shape, dtype=dtype)
        predictions = np.zeros(shape=shape, dtype=dtype)

        labels[0, :2, :2] = 1
        labels[1, 2:, 2:] = 1
        predictions[0, 1:3, 1:3] = 1
        predictions[1, :, 2:] = 1

        ll = sess.run(losses.dice(labels=labels, predictions=predictions, axis=(1, 2), reduction='none'))
        assert np.allclose(ll[0], 0.75)
        assert np.allclose(ll[1], 0.3333333333)
        assert np.allclose(sess.run(losses.dice(labels=labels, predictions=predictions, axis=(1, 2))), 0.5416667)

        zeros = np.zeros((2, 2), dtype=np.float32)
        ones = np.ones((2, 2), dtype=np.float32)

        # Perfect (all zero).
        ll = sess.run(losses.dice(labels=zeros, predictions=zeros, axis=(1,)))
        assert np.allclose(ll, 0)

        # Perfect (all one).
        ll = sess.run(losses.dice(labels=ones, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 0)

        # All wrong.
        ll = sess.run(losses.dice(labels=zeros, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 1)


def test_hamming():
    with tf.Session() as sess:
        zeros = np.zeros((2, 2, 2), dtype=np.float32)
        ones = np.ones((2, 2, 2), dtype=np.float32)

        # Perfect (all zero).
        ll = sess.run(losses.hamming(labels=zeros, predictions=zeros, axis=(1,)))
        assert np.allclose(ll, 0)

        # Perfect (all one).
        ll = sess.run(losses.hamming(labels=ones, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 0)

        # All wrong.
        ll = sess.run(losses.hamming(labels=zeros, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 1)

        # Half of the segmentation classes correct.
        half = zeros.copy()
        half[..., 0] = 1.
        ll = sess.run(losses.hamming(labels=half, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 0.5)

        # 75% wrong
        ones = np.ones((4,))
        preds = np.array([0, 0, 0, 1])
        ll = sess.run(losses.hamming(labels=ones, predictions=preds, axis=(0,), reduction='none'))
        assert np.allclose(ll, 0.75)


def test_tversky():
    with tf.Session() as sess:
        zeros = np.zeros((2, 2, 2), dtype=np.float32)
        ones = np.ones((2, 2, 2), dtype=np.float32)

        # Perfect (all zero).
        ll = sess.run(losses.tversky(labels=zeros, predictions=zeros, axis=(1,)))
        assert np.allclose(ll, 0)

        # Perfect (all one).
        ll = sess.run(losses.tversky(labels=ones, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 0)

        # All wrong.
        ll = sess.run(losses.tversky(labels=zeros, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 1)

        # Half of the segmentation classes correct.
        half = zeros.copy()
        half[..., 0] = 1.
        ll = sess.run(losses.tversky(labels=half, predictions=ones, axis=(1,)))
        assert np.allclose(ll, 0.5)
