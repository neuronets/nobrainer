import itertools

import numpy as np
from numpy.testing import assert_array_equal
import pytest

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
