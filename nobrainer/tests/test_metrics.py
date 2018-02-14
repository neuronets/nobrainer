"""Tests for `nobrainer.metrics`."""

import numpy as np

import nobrainer


def test_dice_coefficient():
    shape = (2, 2, 2)
    x1 = np.ones(shape)
    x2 = np.ones(shape)

    dice = nobrainer.metrics.dice_coefficient_numpy(x1, x2, reducer=None)
    assert np.array_equal(dice, [1, 1])

    dice = nobrainer.metrics.dice_coefficient_numpy(x1, x2, reducer=np.mean)
    assert dice == 1

    x1[0, :, 0] = 0
    dice = nobrainer.metrics.dice_coefficient_numpy(x1, x2, reducer=None)
    assert np.array_equal(dice, [2 / 3, 1])

    dice = nobrainer.metrics.dice_coefficient_numpy(x1, x2, reducer=np.mean)
    assert np.array_equal(dice, np.mean((2 / 3, 1)))
