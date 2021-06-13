import numpy as np
import pytest
import tensorflow as tf

from nobrainer import intensity_transforms


@pytest.fixture(scope="session")
def test_addGaussianNoise():
    shape = (10, 10, 10)
    x = np.ones(shape).astype(np.float32)
    y = np.random.randint(0, 2, size=shape).astype(np.float32)
    x_out = intensity_transforms.addGaussianNoise(x, noise_mean=0.0, noise_std=1)
    x_out = x_out.numpy()
    assert x_out.shape == x.shape
    assert np.sum(x_out - x) != 0

    # test if x and y undergoes same noiseshift
    x_out, y_out = intensity_transforms.addGaussianNoise(
        x, y, trans_xy=True, noise_mean=0.0, noise_std=1
    )
    x_out = x_out.numpy()
    y_out = y_out.numpy()
    noise_y = y_out - y
    noise_x = x_out - x
    assert x_out.shape == x.shape
    assert y_out.shape == y.shape
    assert np.sum(noise_x - noise_y) < 1e-5


def test_minmaxIntensityScaling():
    x = np.random.rand(10, 10, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.float32)
    x, y = intensity_transforms.minmaxIntensityScaling(x, y, trans_xy=True)
    x_out = x.numpy()
    y_out = y.numpy()
    assert x_out.min() - 0.0 < 1e-5
    assert y_out.min() - 0.0 < 1e-5
    assert 1 - x_out.max() < 1e-5
    assert 1 - y_out.max() < 1e-5


def test_customIntensityScaling():
    x = np.random.rand(10, 10, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.float32)
    x, y = intensity_transforms.customIntensityScaling(
        x, y, trans_xy=True, scale_x=[0, 100], scale_y=[0, 3]
    )
    x_out = x.numpy()
    y_out = y.numpy()
    assert x_out.min() - 0.0 < 1e-5
    assert y_out.min() - 0.0 < 1e-5
    assert 100 - x_out.max() < 1e-5
    assert 3 - y_out.max() < 1e-5


def test_intensityMasking():
    mask_x = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])
    expected = np.array(
        [[[0, 0, 0], [0, 2, 0], [0, 0, 0]], [[0, 0, 0], [0, 5, 0], [0, 0, 0]]]
    )
    results = intensity_transforms.intensityMasking(x, mask_x=mask_x)
    results = tf.squeeze(results)
    np.testing.assert_allclose(results.numpy(), expected)


def test_contrastAdjust():
    gamma = 1.5
    epsilon = 1e-7
    x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])
    x_range = x.max() - x.min()
    expected = (
        np.power(((x - x.min()) / float(x_range + epsilon)), gamma) * x_range + x.min()
    )
    results = intensity_transforms.contrastAdjust(x, gamma=1.5)
    np.testing.assert_allclose(expected, results.numpy(), rtol=1e-05)
