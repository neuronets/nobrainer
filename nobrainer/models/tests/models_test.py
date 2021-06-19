import numpy as np
import pytest
import tensorflow as tf

from ..autoencoder import autoencoder
from ..bayesian_vnet import bayesian_vnet
from ..bayesian_vnet_semi import bayesian_vnet_semi
from ..highresnet import highresnet
from ..meshnet import meshnet
from ..progressivegan import progressivegan
from ..unet import unet
from ..vnet import vnet


def model_test(model_cls, n_classes, input_shape, kwds={}):
    """Tests for models."""
    x = 10 * np.random.random(input_shape)
    y = np.random.choice([True, False], input_shape)

    # Assume every model class has n_classes and input_shape arguments.
    model = model_cls(n_classes=n_classes, input_shape=input_shape[1:], **kwds)
    model.compile(tf.optimizers.Adam(), "binary_crossentropy")
    model.fit(x, y)

    actual_output = model.predict(x)
    assert actual_output.shape == x.shape[:-1] + (n_classes,)


def test_highresnet():
    model_test(highresnet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def test_meshnet():
    model_test(
        meshnet,
        n_classes=1,
        input_shape=(1, 32, 32, 32, 1),
        kwds={"receptive_field": 37},
    )
    model_test(
        meshnet,
        n_classes=1,
        input_shape=(1, 32, 32, 32, 1),
        kwds={"receptive_field": 67},
    )
    model_test(
        meshnet,
        n_classes=1,
        input_shape=(1, 32, 32, 32, 1),
        kwds={"receptive_field": 129},
    )
    with pytest.raises(ValueError):
        model_test(
            meshnet,
            n_classes=1,
            input_shape=(1, 32, 32, 32, 1),
            kwds={"receptive_field": 50},
        )


def test_unet():
    model_test(unet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def test_autoencoder():
    """Special test for autoencoder."""

    input_shape = (1, 32, 32, 32, 1)
    x = 10 * np.random.random(input_shape)

    model = autoencoder(input_shape[1:], encoding_dim=128, n_base_filters=32)
    model.compile(tf.optimizers.Adam(), "mse")
    model.fit(x, x)

    actual_output = model.predict(x)
    assert actual_output.shape == x.shape


def test_progressivegan():
    """Test for both discriminator and generator of progressive gan"""

    latent_size = 256
    label_size = 2
    g_fmap_base = 1024
    d_fmap_base = 1024
    alpha = 1.0

    generator, discriminator = progressivegan(
        latent_size,
        label_size=label_size,
        g_fmap_base=g_fmap_base,
        d_fmap_base=d_fmap_base,
    )

    resolutions = [8, 16]

    for res in resolutions:
        generator.add_resolution()
        discriminator.add_resolution()

        latent_input = np.random.random((10, latent_size))
        real_image_input = np.random.random((10, res, res, res, 1))

        fake_images = generator([latent_input, alpha])
        real_pred, real_labels_pred = discriminator([real_image_input, alpha])
        fake_pred, fake_labels_pred = discriminator([fake_images, alpha])

        assert fake_images.shape == real_image_input.shape
        assert real_pred.shape == (real_image_input.shape[0],)
        assert fake_pred.shape == (real_image_input.shape[0],)
        assert real_labels_pred.shape == (real_image_input.shape[0], label_size)
        assert fake_labels_pred.shape == (real_image_input.shape[0], label_size)


def test_vnet():
    model_test(vnet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def test_bayesian_vnet_semi():
    model_test(bayesian_vnet_semi, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def test_bayesian_vnet():
    model_test(bayesian_vnet, n_classes=1, input_shape=(1, 32, 32, 32, 1))
