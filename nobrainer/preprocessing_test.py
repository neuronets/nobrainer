"""Tests for `nobrainer.preprocessing`."""

import numpy as np
from numpy.testing import assert_array_equal

from nobrainer.preprocessing import as_blocks
from nobrainer.preprocessing import binarize
from nobrainer.preprocessing import from_blocks
from nobrainer.preprocessing import normalize_zero_one
from nobrainer.preprocessing import preprocess_aparcaseg
from nobrainer.preprocessing import replace


def test_as_blocks():
    shape = (20, 20, 20)
    data = np.ones(shape)
    blocks = as_blocks(data, (10, 10, 10))
    assert blocks.shape == (8, 10, 10, 10)

    data = np.arange(2**3)
    blocks = as_blocks(data.reshape(2, 2, 2), (1, 1, 1))
    reference = data[..., None, None, None]
    assert_array_equal(blocks, reference)

    shape = (256, 256, 200)
    data = np.ones(shape)
    blocks = as_blocks(data, (128, 128, 100))
    assert blocks.shape == (8, 128, 128, 100)


def test_binarize():
    data = [0, 1, 2, 3, 4]
    assert_array_equal(binarize(data), [0, 1, 1, 1, 1])
    assert_array_equal(binarize(data, threshold=1), [0, 0, 1, 1, 1])
    assert_array_equal(binarize(data, threshold=3), [0, 0, 0, 0, 1])
    assert_array_equal(binarize(data, threshold=1, upper=4), [0, 0, 4, 4, 4])
    assert_array_equal(
        binarize(data, threshold=3, upper=9, lower=8), [8, 8, 8, 8, 9])

    data = np.arange(100)
    data = binarize(data, upper=4, lower=1)
    assert_array_equal(np.unique(data), (1, 4))

    data = np.arange(100)
    data_c = binarize(data)
    assert_array_equal(np.unique(data_c), (0, 1))
    assert not np.array_equal(np.unique(data), (0, 1))


def test_from_blocks():
    data = np.ones((256, 256, 256))
    blocks = as_blocks(data, (128, 128, 128))
    assert_array_equal(data, from_blocks(blocks, (256, 256, 256)))

    data = np.arange(12**3).reshape(12, 12, 12)
    blocks = as_blocks(data, (4, 4, 4))
    assert_array_equal(data, from_blocks(blocks, (12, 12, 12)))


def test_normalize_zero_one():
    data = np.arange(5)
    assert_array_equal(normalize_zero_one(data), [0, 0.25, 0.5, 0.75, 1])

    data = np.random.randint(0, 100, size=100)
    data_norm = normalize_zero_one(data)
    assert data_norm.min() == 0
    assert data_norm.max() == 1


def test_preprocess_aparcaseg():
    mapping = {
        0: 0,
        20: 1,
        30: 2,
    }
    data = np.arange(4) * 10
    preprocess_aparcaseg(data, mapping)
    assert_array_equal(data, [0, 0, 1, 2])
    assert_array_equal(np.unique(data), list(mapping.values()))


def test_replace():
    data = np.arange(5)
    mapping = {
        0: 10,
        1: 20,
        2: 30,
        3: 40,
        4: 30,
    }
    data = replace(data, mapping)
    assert_array_equal(data, [10, 20, 30, 40, 30])

    # Test that overlapping keys and values gives correct result.
    data = np.arange(5)
    mapping = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
    }
    data = replace(data, mapping)
    assert_array_equal(data, [1, 2, 3, 4, 4])
