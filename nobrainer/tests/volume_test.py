import warnings

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from .. import volume


@pytest.mark.parametrize("shape", [(10, 10, 10), (10, 10, 10, 3)])
@pytest.mark.parametrize("scalar_labels", [True, False])
def test_apply_random_transform(shape, scalar_labels):
    x = np.ones(shape).astype(np.float32)
    if scalar_labels:
        transform_func = volume.apply_random_transform_scalar_labels
        y_shape = (1,)
    else:
        transform_func = volume.apply_random_transform
        y_shape = shape

    y_in = np.random.randint(0, 2, size=y_shape).astype(np.float32)
    x, y = transform_func(x, y_in)
    x = x.numpy()
    y = y.numpy()

    # Test that values were not changed in the labels.
    if scalar_labels:
        assert_array_equal(y, y_in)
    else:
        assert_array_equal(np.unique(y), [0, 1])
    assert x.shape == shape
    assert y.shape == y_shape

    with pytest.raises(ValueError):
        inconsistent_shape = tuple([sh + 1 for sh in shape])
        x, y = transform_func(np.ones(shape), np.ones(inconsistent_shape))

    with pytest.raises(ValueError):
        y_shape = (1,) if scalar_labels else (10, 10)
        x, y = transform_func(np.ones((10, 10)), np.ones(y_shape))

    x = np.random.randn(*shape).astype(np.float32)
    y_shape = (1,) if scalar_labels else shape
    y = np.random.randint(0, 2, size=y_shape).astype(np.float32)
    x0, y0 = transform_func(x, y)
    x1, y1 = transform_func(x, y)
    assert not np.array_equal(x, x0)
    assert not np.array_equal(x, x1)
    assert not np.array_equal(x0, x1)

    if scalar_labels:
        assert np.array_equal(y, y0)
        assert np.array_equal(y, y1)
        assert np.array_equal(y0, y1)
    else:
        assert not np.array_equal(y, y0)
        assert not np.array_equal(y, y1)
        assert not np.array_equal(y0, y1)

    # Test that new iterations yield different augmentations.
    x = np.arange(64).reshape(1, 4, 4, 4).astype(np.float32)
    y_shape = (1, 1) if scalar_labels else x.shape
    y = np.random.randint(0, 2, size=y_shape).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # sanity check
    x0, y0 = next(iter(dataset))
    x1, y1 = next(iter(dataset))
    assert_array_equal(x[0], x0)
    assert_array_equal(x0, x1)
    assert_array_equal(y[0], y0)
    assert_array_equal(y0, y1)
    # Need to reset the seed, because it is set in other tests.
    tf.random.set_seed(None)
    dataset = dataset.map(transform_func)
    x0, y0 = next(iter(dataset))
    x1, y1 = next(iter(dataset))
    assert not np.array_equal(x0, x1)
    if scalar_labels:
        assert_array_equal(y0, y1)
    else:
        assert not np.array_equal(y0, y1)
        assert_array_equal(np.unique(y0), [0, 1])
        assert_array_equal(np.unique(y1), [0, 1])
    # Naive test that features were interpolated without nearest neighbor.
    assert np.any(x0 % 1)
    assert np.any(x1 % 1)

    # Depreacation warning test
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        transform_func(x, y_in)

        assert len(w) == 1
        assert issubclass(w[-1].category, PendingDeprecationWarning)
        assert "moved" in str(w[-1].message)


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
