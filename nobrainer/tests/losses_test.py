from distutils.version import LooseVersion

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy.spatial.distance
import tensorflow as tf

from .. import losses


def test_dice():
    x = np.zeros(4)
    y = np.zeros(4)
    out = losses.dice(x, y, axis=None).numpy()
    assert_allclose(out, 0)

    x = np.ones(4)
    y = np.ones(4)
    out = losses.dice(x, y, axis=None).numpy()
    assert_allclose(out, 0)

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0, 1.0]
    out = losses.dice(x, y, axis=None).numpy()
    ref = scipy.spatial.distance.dice(x, y)
    assert_allclose(out, ref)

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 0.0, 0.0]
    out = losses.dice(x, y, axis=None).numpy()
    ref = scipy.spatial.distance.dice(x, y)
    assert_allclose(out, ref)
    assert_allclose(out, 1)

    x = np.ones((4, 32, 32, 32, 1), dtype=np.float32)
    y = x.copy()
    x[:2, :10, 10:] = 0
    y[:2, :3, 20:] = 0
    y[3:, 10:] = 0
    dices = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        dices[i] = scipy.spatial.distance.dice(x[i].flatten(), y[i].flatten())
    assert_allclose(losses.dice(x, y, axis=(1, 2, 3, 4)), dices, rtol=1e-05)
    assert_allclose(losses.Dice(axis=(1, 2, 3, 4))(x, y), dices.mean(), rtol=1e-05)
    assert_allclose(losses.Dice(axis=(1, 2, 3, 4))(y, x), dices.mean(), rtol=1e-05)


def test_generalized_dice():
    shape = (8, 32, 32, 32, 16)
    x = np.zeros(shape)
    y = np.zeros(shape)
    assert_array_equal(losses.generalized_dice(x, y), np.zeros(shape[0]))

    shape = (8, 32, 32, 32, 16)
    x = np.ones(shape)
    y = np.ones(shape)
    assert_array_equal(losses.generalized_dice(x, y), np.zeros(shape[0]))

    shape = (8, 32, 32, 32, 16)
    x = np.ones(shape)
    y = np.zeros(shape)
    # Why aren't the losses exactly one? Could it be the propagation of floating
    # point inaccuracies when summing?
    assert_allclose(losses.generalized_dice(x, y), np.ones(shape[0]), atol=1e-03)
    assert_allclose(
        losses.GeneralizedDice(axis=(1, 2, 3))(x, y), losses.generalized_dice(x, y)
    )

    x = np.ones((4, 32, 32, 32, 1), dtype=np.float64)
    y = x.copy()
    x[:2, :10, 10:] = 0
    y[:2, :3, 20:] = 0
    y[3:, 10:] = 0
    # Dice is similar to generalized Dice for one class. The weight factor
    # makes the generalized form slightly different from Dice.
    gd = losses.generalized_dice(x, y, axis=(1, 2, 3)).numpy()
    dd = losses.dice(x, y, axis=(1, 2, 3, 4)).numpy()
    assert_allclose(gd, dd, rtol=1e-02)  # is this close enough?


def test_jaccard():
    x = np.zeros(4)
    y = np.zeros(4)
    out = losses.jaccard(x, y, axis=None).numpy()
    assert_allclose(out, 0)

    x = np.ones(4)
    y = np.ones(4)
    out = losses.jaccard(x, y, axis=None).numpy()
    assert_allclose(out, 0)

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0, 1.0]
    out = losses.jaccard(x, y, axis=None).numpy()
    ref = scipy.spatial.distance.jaccard(x, y)
    assert_allclose(out, ref)

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 0.0, 0.0]
    out = losses.jaccard(x, y, axis=None).numpy()
    ref = scipy.spatial.distance.jaccard(x, y)
    assert_allclose(out, ref)
    assert_allclose(out, 1)

    x = np.ones((4, 32, 32, 32, 1), dtype=np.float32)
    y = x.copy()
    x[:2, :10, 10:] = 0
    y[:2, :3, 20:] = 0
    y[3:, 10:] = 0
    jaccards = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        jaccards[i] = scipy.spatial.distance.jaccard(x[i].flatten(), y[i].flatten())
    assert_allclose(losses.jaccard(x, y, axis=(1, 2, 3, 4)), jaccards)
    assert_allclose(losses.Jaccard(axis=(1, 2, 3, 4))(x, y), jaccards.mean())
    assert_allclose(losses.Jaccard(axis=(1, 2, 3, 4))(y, x), jaccards.mean())


@pytest.mark.xfail
def test_tversky():
    # TODO: write the test
    assert False


@pytest.mark.xfail
def test_elbo():
    # TODO: write the test
    assert False


def test_wasserstein():
    x = np.zeros(4)
    y = np.zeros(4)
    out = losses.wasserstein(x, y)
    assert_allclose(out, 0)

    x = np.ones(4)
    y = np.ones(4)
    out = losses.wasserstein(x, y)
    assert_allclose(out, 1)

    x = np.array([0.0, -1.0, 1.0, -1.0])
    y = np.array([1.0, -1.0, 1.0, 1.0])
    out = losses.wasserstein(x, y)
    ref = [0.0, 1.0, 1.0, -1.0]
    assert_allclose(out, ref)

    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    out = losses.wasserstein(x, y)
    assert_allclose(out, 0)


def test_gradient_penalty():
    x = np.zeros(4)
    y = np.zeros(4)
    out = losses.gradient_penalty(x, y)
    assert_allclose(out, 10)

    x = np.ones(4)
    y = np.ones(4)
    out = losses.gradient_penalty(x, y)
    assert_allclose(out, 0.001)

    x = np.array([0.0, -1.0, 1.0, -1.0])
    y = np.array([1.0, -1.0, 1.0, 1.0])
    out = losses.gradient_penalty(x, y)
    ref = [1.0001e01, 1.0000e-03, 1.0000e-03, 1.0000e-03]
    assert_allclose(out, ref)

    x = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    out = losses.gradient_penalty(x, y)
    ref = [10.001, 10.001, 0.0, 0.0]
    assert_allclose(out, ref)


def test_get():
    if LooseVersion(tf.__version__) < LooseVersion("1.14.1-dev20190408"):
        assert losses.get("dice") is losses.dice
        assert losses.get("Dice") is losses.Dice
        assert losses.get("jaccard") is losses.jaccard
        assert losses.get("Jaccard") is losses.Jaccard
        assert losses.get("tversky") is losses.tversky
        assert losses.get("Tversky") is losses.Tversky
        assert losses.get("binary_crossentropy")
    else:
        assert losses.get("dice") is losses.dice
        assert isinstance(losses.get("Dice"), losses.Dice)
        assert losses.get("jaccard") is losses.jaccard
        assert isinstance(losses.get("Jaccard"), losses.Jaccard)
        assert losses.get("tversky") is losses.tversky
        assert isinstance(losses.get("Tversky"), losses.Tversky)
        assert losses.get("binary_crossentropy")
        assert losses.get("gradient_penalty") is losses.gradient_penalty
        assert losses.get("wasserstein") is losses.wasserstein
        assert isinstance(losses.get("Wasserstein"), losses.Wasserstein)
