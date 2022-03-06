import itertools

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from .. import transform


@pytest.mark.parametrize("volume_shape", [(64, 64, 64), (64, 64, 64, 3)])
def test_get_affine_smoke(volume_shape):
    affine = transform.get_affine(volume_shape)

    assert_array_equal(affine, np.eye(4))


def test_get_affine_errors():
    with pytest.raises(ValueError):
        transform.get_affine(volume_shape=(64, 64))

    with pytest.raises(ValueError):
        transform.get_affine(volume_shape=(64, 64, 64), rotation=[0, 0])

    with pytest.raises(ValueError):
        transform.get_affine(volume_shape=(64, 64, 64), translation=[0, 0])


@pytest.mark.parametrize("volume_shape", [(2, 2, 2), (2, 2, 2, 3)])
def test_get_coordinates(volume_shape):
    coords = transform._get_coordinates(volume_shape=volume_shape)
    coords_ref = [
        list(element) for element in list(itertools.product([0, 1], repeat=3))
    ]
    assert_array_equal(coords, coords_ref)


def test_get_coordinates_errors():
    with pytest.raises(ValueError):
        transform._get_coordinates(volume_shape=(64, 64))


@pytest.mark.parametrize("volume_shape", [(8, 8, 8), (8, 8, 8, 3)])
def test_trilinear_interpolation_smoke(volume_shape):
    volume = np.arange(np.prod(volume_shape)).reshape(volume_shape)
    coords = transform._get_coordinates(volume_shape=volume_shape)
    x = transform._trilinear_interpolation(volume=volume, coords=coords)
    assert_array_equal(x, volume)


@pytest.mark.parametrize("volume_shape", [(8, 8, 8), (8, 8, 8, 3)])
def test_get_voxels(volume_shape):
    volume = np.arange(np.prod(volume_shape)).reshape(volume_shape)
    coords = transform._get_coordinates(volume_shape=volume_shape)
    voxels = transform._get_voxels(volume=volume, coords=coords)

    if len(volume_shape) == 3:
        assert_array_equal(voxels, np.arange(np.prod(volume_shape)))
    else:
        assert_array_equal(
            voxels,
            np.arange(np.prod(volume_shape)).reshape((np.prod(volume_shape[:3]), -1)),
        )


def test_get_voxels_errors():
    volume = np.zeros((8, 8))
    coords = transform._get_coordinates(volume_shape=(8, 8, 8))
    with pytest.raises(ValueError):
        transform._get_voxels(volume=volume, coords=coords)

    volume = np.zeros((8, 8, 8))
    coords = np.zeros((8, 8, 8))
    with pytest.raises(ValueError):
        transform._get_voxels(volume=volume, coords=coords)

    coords = np.zeros((8, 2))
    with pytest.raises(ValueError):
        transform._get_voxels(volume=volume, coords=coords)


@pytest.mark.parametrize("shape", [(10, 10, 10), (10, 10, 10, 3)])
@pytest.mark.parametrize("scalar_labels", [True, False])
def test_apply_random_transform(shape, scalar_labels):
    x = np.ones(shape).astype(np.float32)
    transform_func = transform.apply_random_transform
    if scalar_labels:
        y_shape = (1,)
        kwargs = {"trans_xy": False}
    else:
        y_shape = shape
        kwargs = {}

    y_in = np.random.randint(0, 2, size=y_shape).astype(np.float32)
    x, y = transform_func(x, y_in, **kwargs)
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
        x, y = transform_func(np.ones(shape), np.ones(inconsistent_shape), **kwargs)

    with pytest.raises(ValueError):
        y_shape = (1,) if scalar_labels else (10, 10)
        x, y = transform_func(np.ones((10, 10)), np.ones(y_shape), **kwargs)

    x = np.random.randn(*shape).astype(np.float32)
    y_shape = (1,) if scalar_labels else shape
    y = np.random.randint(0, 2, size=y_shape).astype(np.float32)
    x0, y0 = transform_func(x, y, **kwargs)
    x1, y1 = transform_func(x, y, **kwargs)
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
    dataset = dataset.map(lambda x_l, y_l: transform_func(x_l, y_l, **kwargs))
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
