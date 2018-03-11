"""Tests for `nobrainer.metrics`."""

import numpy as np
import scipy as sp
import tensorflow as tf

from nobrainer.metrics import dice, dice_numpy


def test_dice():
    shape = (2, 10)

    foo = np.zeros(shape, dtype=np.float64)
    foo[:, 4:] = 1

    bar = np.zeros(shape, dtype=np.float64)
    bar[0, :7] = 1
    bar[1, :5] = 1

    true_dices = np.zeros(foo.shape[0])
    for idx in range(foo.shape[0]):
        true_dices[idx] = sp.spatial.distance.dice(
            foo[idx].flatten(), bar[idx].flatten())

    with tf.Session() as sess:
        u_ = tf.placeholder(tf.float64)
        v_ = tf.placeholder(tf.float64)
        dice_coeffs = dice(u_, v_, axis=-1)

        test_dices = sess.run(dice_coeffs, feed_dict={u_: foo, v_: bar})

    # Test TensorFlow implementation.
    np.testing.assert_almost_equal(1 - test_dices, true_dices)

    # Test NumPy implementation.
    test_dices_np = dice_numpy(foo, bar, axis=-1)
    np.testing.assert_almost_equal(1 - test_dices_np, true_dices)


def test_hamming():
    shape = (2, 10)

    foo = np.zeros(shape, dtype=np.float64)
    foo[:, 4:] = 1

    bar = np.zeros(shape, dtype=np.float64)
    bar[0, :7] = 1
    bar[1, :5] = 1

    true_hammings = np.zeros(foo.shape[0])
    for idx in range(foo.shape[0]):
        true_hammings[idx] = sp.spatial.distance.hamming(
            foo[idx].flatten(), bar[idx].flatten())

    with tf.Session() as sess:
        u_ = tf.placeholder(tf.float64)
        v_ = tf.placeholder(tf.float64)
        test_hammings = sess.run(
            dice(u_, v_, axis=-1), feed_dict={u_: foo, v_: bar})

    # Test TensorFlow implementation.
    np.testing.assert_almost_equal(test_hammings, true_hammings)

    # Test NumPy implementation.
    test_hammings_np = dice_numpy(foo, bar, axis=-1)
    np.testing.assert_almost_equal(test_hammings_np, true_hammings)
