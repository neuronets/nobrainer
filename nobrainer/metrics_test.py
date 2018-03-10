"""Tests for `nobrainer.metrics`."""

import numpy as np

from nobrainer.metrics import dice_coefficient_numpy


def test_dice_coefficient_numpy():
    shape = (2, 2, 2)
    x1 = np.ones(shape)
    x2 = np.ones(shape)

    dice = dice_coefficient_numpy(x1, x2, reducer=None)
    assert np.array_equal(dice, [1, 1])

    dice = dice_coefficient_numpy(x1, x2, reducer=np.mean)
    assert dice == 1

    x1[0, :, 0] = 0
    dice = dice_coefficient_numpy(x1, x2, reducer=None)
    assert np.array_equal(dice, [2 / 3, 1])

    dice = dice_coefficient_numpy(x1, x2, reducer=np.mean)
    assert np.array_equal(dice, np.mean((2 / 3, 1)))

    x1 = np.zeros(shape)
    x2 = np.zeros(shape)
    assert np.isnan(dice_coefficient_numpy(x1, x2))
