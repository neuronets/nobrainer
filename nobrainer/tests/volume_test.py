import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from .. import volume


def test_binarize():
    x = [
        0.49671415,
        -0.1382643,
        0.64768854,
        1.52302986,
        -0.23415337,
        -0.23413696,
        1.57921282,
        0.76743473,
    ]
    x = np.asarray(x, dtype="float64")
    expected = np.array([True, False, True, True, False, False, True, True])
    result = volume.binarize(x)
    assert_array_equal(expected, result)
    assert result.dtype == tf.float64
    result = volume.binarize(x.astype(np.float32))
    assert_array_equal(expected, result)
    assert result.dtype == tf.float32

    x = np.asarray([-2, 0, 2, 0, 2, -2, -1, 1], dtype=np.int32)
    expected = np.array([False, False, True, False, True, False, False, True])
    result = volume.binarize(x)
    assert_array_equal(expected, result)
    assert result.dtype == tf.int32
    result = volume.binarize(x.astype(np.int64))
    assert_array_equal(expected, result)
    assert result.dtype == tf.int64


@pytest.mark.parametrize("replace_func", [volume.replace, volume.replace_in_numpy])
def test_replace(replace_func):
    data = np.arange(5)
    mapping = {0: 10, 1: 20, 2: 30, 3: 40, 4: 30}
    output = replace_func(data, mapping)
    assert_array_equal(output, [10, 20, 30, 40, 30])

    # Test that overlapping keys and values gives correct result.
    data = np.arange(5)
    mapping = {0: 1, 1: 2, 2: 3, 3: 4}
    output = replace_func(data, mapping)
    assert_array_equal(output, [1, 2, 3, 4, 0])

    data = np.arange(8).reshape(2, 2, 2)
    mapping = {0: 100, 100: 10, 10: 5, 3: 5}
    outputs = replace_func(data, mapping, zero=False)
    expected = data.copy()
    expected[0, 0, 0] = 100
    expected[0, 1, 1] = 5
    assert_array_equal(outputs, expected)

    # Zero values not in mapping values.
    outputs = replace_func(data, mapping, zero=True)
    expected = np.zeros_like(data)
    expected[0, 0, 0] = 100
    expected[0, 1, 1] = 5
    assert_array_equal(outputs, expected)


@pytest.mark.parametrize("std_func", [volume.standardize, volume.standardize_numpy])
def test_standardize(std_func):
    x = np.random.randn(10, 10, 10).astype(np.float32)
    outputs = np.array(std_func(x))
    assert np.allclose(outputs.mean(), 0, atol=1e-07)
    assert np.allclose(outputs.std(), 1, atol=1e-07)

    if std_func == volume.standardize:
        x = np.random.randn(10, 10, 10).astype(np.float64)
        outputs = np.array(std_func(x))
        assert outputs.dtype == np.float32


@pytest.mark.parametrize("norm_func", [volume.normalize, volume.normalize_numpy])
def test_normalize(norm_func):
    x = np.random.randn(10, 10, 10).astype(np.float32)
    outputs = np.array(norm_func(x))
    assert np.allclose(outputs.min(), 0, atol=1e-07)
    assert np.allclose(outputs.max(), 1, atol=1e-07)

    if norm_func == volume.normalize:
        x = np.random.randn(10, 10, 10).astype(np.float64)
        outputs = np.array(norm_func(x))
        assert outputs.dtype == np.float32


def _stack_channels(_in):
    return np.stack([_in, 2 * _in, 3 * _in], axis=-1)


@pytest.mark.parametrize("multichannel", [True, False])
@pytest.mark.parametrize("to_blocks_func", [volume.to_blocks, volume.to_blocks_numpy])
def test_to_blocks(multichannel, to_blocks_func):
    x = np.arange(8).reshape(2, 2, 2)
    block_shape = (1, 1, 1)
    if multichannel:
        x = _stack_channels(x)
        block_shape = (1, 1, 1, 3)
    outputs = np.array(to_blocks_func(x, block_shape))
    expected = np.array(
        [[[[0]]], [[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]], [[[6]]], [[[7]]]]
    )
    if multichannel:
        expected = _stack_channels(expected)
    assert_array_equal(outputs, expected)

    block_shape = 2
    if multichannel:
        block_shape = (2, 2, 2, 3)
    outputs = np.array(to_blocks_func(x, block_shape))
    assert_array_equal(outputs, x[None])

    block_shape = (3, 3, 3)
    if multichannel:
        block_shape = (3, 3, 3, 3)
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        to_blocks_func(x, block_shape)

    block_shape = (3, 3)
    with pytest.raises(ValueError):
        to_blocks_func(x, block_shape)


@pytest.mark.parametrize("multichannel", [True, False])
@pytest.mark.parametrize(
    "from_blocks_func", [volume.from_blocks, volume.from_blocks_numpy]
)
def test_from_blocks(multichannel, from_blocks_func):
    x = np.arange(64).reshape(4, 4, 4)
    block_shape = (2, 2, 2)
    if multichannel:
        x = _stack_channels(x)
        block_shape = (2, 2, 2, 3)

    outputs = from_blocks_func(volume.to_blocks(x, block_shape), x.shape)
    assert_array_equal(outputs, x)

    with pytest.raises(ValueError):
        x = np.arange(80).reshape(10, 2, 2, 2)
        outputs = from_blocks_func(x, (4, 4, 4))


def test_blocks_numpy_value_errors():
    with pytest.raises(ValueError):
        x = np.random.rand(4, 4)
        output_shape = (4, 4, 4)
        volume.to_blocks_numpy(x, output_shape)

    with pytest.raises(ValueError):
        x = np.random.rand(4, 4, 4)
        output_shape = (4, 4, 4)
        volume.from_blocks_numpy(x, output_shape)

    with pytest.raises(ValueError):
        x = np.random.rand(4, 4, 4, 4)
        output_shape = (4, 4)
        volume.from_blocks_numpy(x, output_shape)
