import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from nobrainer import volume


@pytest.mark.xfail
def test_get_dataset():
    assert False


@pytest.mark.xfail
def test_apply_random_transform():
    assert False


@pytest.mark.xfail
def test_apply_random_transform_dataset():
    assert False


def test_binarize():
    x = [ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337,
       -0.23413696,  1.57921282,  0.76743473]
    x = np.asarray(x, dtype='float64')
    expected = np.array([ True, False,  True,  True, False, False,  True,  True])
    result = volume.binarize(x)
    assert_array_equal(expected, result)
    assert result.dtype == tf.float64
    result = volume.binarize(x.astype(np.float32))
    assert_array_equal(expected, result)
    assert result.dtype == tf.float32

    x = np.asarray([-2,  0,  2,  0,  2, -2, -1,  1], dtype=np.int32)
    expected = np.array([False, False, True, False, True, False, False, True])
    result = volume.binarize(x)
    assert_array_equal(expected, result)
    assert result.dtype == tf.int32
    result = volume.binarize(x.astype(np.int64))
    assert_array_equal(expected, result)
    assert result.dtype == tf.int64


def test_replace():
    data = np.arange(5)
    mapping = {
        0: 10,
        1: 20,
        2: 30,
        3: 40,
        4: 30,
    }
    output = volume.replace(data, mapping)
    assert_array_equal(output, [10, 20, 30, 40, 30])

    # Test that overlapping keys and values gives correct result.
    data = np.arange(5)
    mapping = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
    }
    output = volume.replace(data, mapping)
    assert_array_equal(output, [1, 2, 3, 4, 4])

    data = np.arange(8).reshape(2, 2, 2)
    mapping = {0: 100, 100: 10, 10: 5, 3: 5}
    outputs = volume.replace(data, mapping, zero=False)
    expected = data.copy()
    expected[0, 0, 0] = 100
    expected[0, 1, 1] = 5
    assert_array_equal(outputs, expected)

    # Zero values not in mapping values.
    outputs = volume.replace(data, mapping, zero=True)
    expected = np.zeros_like(data)
    expected[0, 0, 0] = 100
    expected[0, 1, 1] = 5
    expected[1, 0, 1] = 5
    assert_array_equal(outputs, expected)


def test_standardize():
    x = np.random.randn(10, 10, 10).astype(np.float32)
    outputs = volume.standardize(x).numpy()
    assert np.allclose(outputs.mean(), 0, atol=1e-07)
    assert np.allclose(outputs.std(), 1, atol=1e-07)


def test_to_blocks():
    x = np.arange(8).reshape(2, 2, 2)
    outputs = volume.to_blocks(x, (1, 1, 1)).numpy()
    expected = np.array([[[[0]]],
                       [[[1]]],
                       [[[2]]],
                       [[[3]]],
                       [[[4]]],
                       [[[5]]],
                       [[[6]]],
                       [[[7]]]])
    assert_array_equal(outputs, expected)
    outputs = volume.to_blocks(x, (2, 2, 2)).numpy()
    assert_array_equal(outputs, x[None])

    with pytest.raises(tf.errors.InvalidArgumentError):
        volume.to_blocks(x, (3, 3, 3))


def test_from_blocks():
    x = np.arange(64).reshape(4, 4, 4)
    block_shape = (2, 2, 2)
    outputs = volume.from_blocks(volume.to_blocks(x, block_shape), x.shape)
    assert_array_equal(outputs, x)


@pytest.mark.xfail
def test_get_preprocess_fn():
    assert False


@pytest.mark.xfail
def test_preprocess_binary():
    assert False


@pytest.mark.xfail
def test_preprocess_multiclass():
    assert False


@pytest.mark.xfail
def test_get_steps_per_epoch():
    assert False
