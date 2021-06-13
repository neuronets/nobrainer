import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy.spatial.distance

from .. import metrics


def test_dice():
    x = np.zeros(4)
    y = np.zeros(4)
    out = metrics.dice(x, y, axis=None).numpy()
    assert_allclose(out, 1)

    x = np.ones(4)
    y = np.ones(4)
    out = metrics.dice(x, y, axis=None).numpy()
    assert_allclose(out, 1)

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0, 1.0]
    out = metrics.dice(x, y, axis=None).numpy()
    ref = 1.0 - scipy.spatial.distance.dice(x, y)
    assert_allclose(out, ref)
    jac_out = metrics.jaccard(x, y, axis=None).numpy()
    assert_allclose(out, 2.0 * jac_out / (1.0 + jac_out))

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 0.0, 0.0]
    out = metrics.dice(x, y, axis=None).numpy()
    ref = 1.0 - scipy.spatial.distance.dice(x, y)
    assert_allclose(out, ref, atol=1e-07)
    assert_allclose(out, 0, atol=1e-07)

    x = np.ones((4, 32, 32, 32, 1), dtype=np.float32)
    y = x.copy()
    x[:2, :10, 10:] = 0
    y[:2, :3, 20:] = 0
    y[3:, 10:] = 0
    dices = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        dices[i] = 1.0 - scipy.spatial.distance.dice(x[i].flatten(), y[i].flatten())
    assert_allclose(metrics.dice(x, y, axis=(1, 2, 3, 4)), dices)


def test_generalized_dice():
    shape = (8, 32, 32, 32, 16)
    x = np.zeros(shape)
    y = np.zeros(shape)
    assert_array_equal(metrics.generalized_dice(x, y), np.ones(shape[0]))

    shape = (8, 32, 32, 32, 16)
    x = np.ones(shape)
    y = np.ones(shape)
    assert_array_equal(metrics.generalized_dice(x, y), np.ones(shape[0]))

    shape = (8, 32, 32, 32, 16)
    x = np.ones(shape)
    y = np.zeros(shape)
    # Why aren't the scores exactly zero? Could it be the propagation of floating
    # point inaccuracies when summing?
    assert_allclose(metrics.generalized_dice(x, y), np.zeros(shape[0]), atol=1e-03)

    x = np.ones((4, 32, 32, 32, 1), dtype=np.float64)
    y = x.copy()
    x[:2, :10, 10:] = 0
    y[:2, :3, 20:] = 0
    y[3:, 10:] = 0
    # Dice is similar to generalized Dice for one class. The weight factor
    # makes the generalized form slightly different from Dice.
    gd = metrics.generalized_dice(x, y, axis=(1, 2, 3)).numpy()
    dd = metrics.dice(x, y, axis=(1, 2, 3, 4)).numpy()
    assert_allclose(gd, dd, rtol=1e-02)  # is this close enough?


def test_jaccard():
    x = np.zeros(4)
    y = np.zeros(4)
    out = metrics.jaccard(x, y, axis=None).numpy()
    assert_allclose(out, 1)

    x = np.ones(4)
    y = np.ones(4)
    out = metrics.jaccard(x, y, axis=None).numpy()
    assert_allclose(out, 1)

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0, 1.0]
    out = metrics.jaccard(x, y, axis=None).numpy()
    ref = 1.0 - scipy.spatial.distance.jaccard(x, y)
    assert_allclose(out, ref)
    dice_out = metrics.dice(x, y, axis=None).numpy()
    assert_allclose(out, dice_out / (2.0 - dice_out))

    x = [0.0, 0.0, 1.0, 1.0]
    y = [1.0, 1.0, 0.0, 0.0]
    out = metrics.jaccard(x, y, axis=None).numpy()
    ref = 1.0 - scipy.spatial.distance.jaccard(x, y)
    assert_allclose(out, ref, atol=1e-07)
    assert_allclose(out, 0, atol=1e-07)

    x = np.ones((4, 32, 32, 32, 1), dtype=np.float32)
    y = x.copy()
    x[:2, :10, 10:] = 0
    y[:2, :3, 20:] = 0
    y[3:, 10:] = 0
    jaccards = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        jaccards[i] = 1.0 - scipy.spatial.distance.jaccard(
            x[i].flatten(), y[i].flatten()
        )
    assert_allclose(metrics.jaccard(x, y, axis=(1, 2, 3, 4)), jaccards)


def test_tversky():
    shape = (4, 32, 32, 32, 1)
    y_pred = np.random.rand(*shape).astype(np.float64)
    y_true = np.random.randint(2, size=shape).astype(np.float64)

    # Test that tversky and dice are same when alpha = beta = 0.5
    dice = metrics.dice(y_true, y_pred).numpy()
    tversky = metrics.tversky(
        y_true, y_pred, axis=(1, 2, 3), alpha=0.5, beta=0.5
    ).numpy()
    assert_allclose(dice, tversky)

    # Test that tversky and jaccard are same when alpha = beta = 1.0
    jaccard = metrics.jaccard(y_true, y_pred).numpy()
    tversky = metrics.tversky(
        y_true, y_pred, axis=(1, 2, 3), alpha=1.0, beta=1.0
    ).numpy()
    assert_allclose(jaccard, tversky)

    with pytest.raises(ValueError):
        metrics.tversky([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], axis=0)
