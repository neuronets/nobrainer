# -*- coding: utf-8 -*-
"""Tests for `nobrainer.volume` module."""

import numpy as np
from numpy.testing import assert_array_equal

from nobrainer.volume import binarize
from nobrainer.volume import change_brightness
from nobrainer.volume import downsample
from nobrainer.volume import flip
from nobrainer.volume import from_blocks
from nobrainer.volume import iterblocks_3d
from nobrainer.volume import itervolumes
from nobrainer.volume import normalize_zero_one
from nobrainer.volume import reduce_contrast
from nobrainer.volume import replace
from nobrainer.volume import rotate
from nobrainer.volume import salt_and_pepper
from nobrainer.volume import shift
from nobrainer.volume import to_blocks
from nobrainer.volume import zoom
from nobrainer.volume import zscore
from nobrainer.volume import VolumeDataGenerator


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


def test_change_brightness():
    a = np.arange(5)  # 0 1 2 3 4
    assert_array_equal(
        change_brightness(a, 1, lower_clip=2, upper_clip=4),
        [2, 2, 3, 4, 4])
    assert change_brightness(a, 10, out=a) is a


def test_downsample():
    a = np.arange(6)
    assert_array_equal(downsample(a, 2), [0, 0, 2, 2, 4, 4])


def test_flip():
    a = np.arange(6)
    assert_array_equal(flip(a, 0), np.arange(6)[::-1])


def test_from_blocks():
    data = np.ones((256, 256, 256))
    blocks = to_blocks(data, (128, 128, 128))
    assert_array_equal(data, from_blocks(blocks, (256, 256, 256)))

    data = np.arange(12**3).reshape(12, 12, 12)
    blocks = to_blocks(data, (4, 4, 4))
    assert_array_equal(data, from_blocks(blocks, (12, 12, 12)))


def test_iterblocks_3d():
    a = np.arange(8).reshape(2, 2, 2)
    this = next(iterblocks_3d(a, kernel_size=(2, 2, 2)))
    assert_array_equal(this, a)


def test_itervolumes():
    itervolumes


def test_normalize_zero_one():
    data = np.arange(5)
    assert_array_equal(normalize_zero_one(data), [0, 0.25, 0.5, 0.75, 1])
    data = np.random.randint(0, 100, size=100)
    data_norm = normalize_zero_one(data)
    assert data_norm.min() == 0
    assert data_norm.max() == 1


def test_reduce_contrast():
    a = np.arange(100)
    assert_array_equal(reduce_contrast(a), np.sqrt(a))


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


def test_rotate():
    rotate


def test_salt_and_pepper():
    salt_and_pepper


def test_shift():
    shift


def test_to_blocks():
    shape = (20, 20, 20)
    data = np.ones(shape)
    blocks = to_blocks(data, (10, 10, 10))
    assert blocks.shape == (8, 10, 10, 10)

    data = np.arange(2**3)
    blocks = to_blocks(data.reshape(2, 2, 2), (1, 1, 1))
    reference = data[..., None, None, None]
    assert_array_equal(blocks, reference)

    shape = (256, 256, 200)
    data = np.ones(shape)
    blocks = to_blocks(data, (128, 128, 100))
    assert blocks.shape == (8, 128, 128, 100)


def test_zoom():
    zoom


def test_zscore():
    zscore


class TestVolumeDataGenerator:
    def test_init(self):
        VolumeDataGenerator

    def test_standardize(self):
        pass

    def test_random_transform(self):
        pass

    def test_flow_from_filepaths(self):
        pass

    def test_dset_input_fn_builder(self):
        pass


def test_get_n_blocks():
    pass
