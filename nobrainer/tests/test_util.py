"""Tests for `nobrainer.util`."""

import numpy as np
import pytest

from nobrainer import util


def test__shapes_equal():
    shape = (2, 2, 2)
    x1 = np.zeros(shape)
    x2 = np.zeros(shape)

    assert util._shapes_equal(x1, x2)
    assert not util._shapes_equal(x1, x2.reshape(-1))


def test__check_shapes_equal(x1, x2):
    shape = (2, 2, 2)
    x1 = np.zeros(shape)
    x2 = np.zeros(shape)

    util._check_shapes_equal(x1, x2)

    with pytest.raises(ValueError):
        util._check_shapes_equal(x1, x2.reshape(-1))


def test__check_all_x_in_subset_numpy():
    x = np.zeros(10)
    x[:5] = 1

    util._check_all_x_in_subset_numpy(x, (0, 1))
    util._check_all_x_in_subset_numpy(x, (0, 1, 2, 3))

    with pytest.raises(ValueError):
        util._check_all_x_in_subset_numpy(x, (0, 2))

    with pytest.raises(ValueError):
        util._check_all_x_in_subset_numpy(x, (1, 2))
