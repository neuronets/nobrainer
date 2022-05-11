import numpy as np
import pytest
import tensorflow as tf

from nobrainer.bayesian_utils import default_mean_field_normal_fn

from ..autoencoder import autoencoder
from ..bayesian_vnet import bayesian_vnet
from ..bayesian_vnet_semi import bayesian_vnet_semi
from ..brainsiam import brainsiam
from ..dcgan import dcgan
from ..highresnet import highresnet
from ..meshnet import meshnet
from ..progressivegan import progressivegan
from ..unet import unet
from ..unet_lstm import unet_lstm
from ..vnet import vnet
from ..vox2vox import Vox_ensembler, vox_gan


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


def test_brainsiam():
    """Testing the encoder-projector and predictor structures of the brainsiam architecture"""
    input_shape = (1, 32, 32, 32, 1)
    x = 10 * np.random.random(input_shape)

    n_classes = 1
    weight_decay = 0.0005
    projection_dim = 2048
    latent_dim = 512

    encoder, predictor = brainsiam(
        n_classes,
        input_shape=input_shape[1:],
        weight_decay=weight_decay,
        projection_dim=projection_dim,
        latent_dim=latent_dim,
    )

    encoder_output = encoder(x[1:])
    enc_output_shape = encoder_output.get_shape().as_list()

    predictor_out = predictor(encoder_output)
    pred_output_shape = predictor_out.get_shape().as_list()

    assert (
        enc_output_shape[1] == projection_dim
    ), "encoder output shape not the same as projection dim"
    assert (
        pred_output_shape[1] == projection_dim
    ), "predictor output shape not the same as projection dim"


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


def test_dcgan():
    """Special test for dcgan."""

    output_shape = (1, 32, 32, 32, 1)
    z_dim = 32
    z = np.random.random((1, z_dim))

    pred_shape = (1, 8, 8, 8, 1)

    generator, discriminator = dcgan(output_shape[1:], z_dim=z_dim)
    generator.compile(tf.optimizers.Adam(), "mse")
    discriminator.compile(tf.optimizers.Adam(), "mse")

    fake_images = generator.predict(z)
    fake_pred = discriminator.predict(fake_images)

    assert fake_images.shape == output_shape and fake_pred.shape == pred_shape


def test_vnet():
    model_test(vnet, n_classes=1, input_shape=(1, 32, 32, 32, 1))


def model_test_bayesian(model_cls, n_classes, input_shape, kernel_posterior_fn):
    """Tests for models."""
    x = 10 * np.random.random(input_shape)
    y = np.random.choice([True, False], input_shape)

    # Assume every model class has n_classes and input_shape arguments.
    model = model_cls(
        n_classes=n_classes,
        input_shape=input_shape[1:],
        kernel_posterior_fn=kernel_posterior_fn,
    )
    model.compile(tf.optimizers.Adam(), "binary_crossentropy")
    model.fit(x, y)

    actual_output = model.predict(x)
    assert actual_output.shape == x.shape[:-1] + (n_classes,)


def test_bayesian_vnet_semi():
    model_test_bayesian(
        bayesian_vnet_semi,
        n_classes=1,
        input_shape=(1, 32, 32, 32, 1),
        kernel_posterior_fn=default_mean_field_normal_fn(weightnorm=True),
    )


def test_bayesian_vnet():
    model_test_bayesian(
        bayesian_vnet,
        n_classes=1,
        input_shape=(1, 32, 32, 32, 1),
        kernel_posterior_fn=default_mean_field_normal_fn(weightnorm=True),
    )


def test_unet_lstm():
    input_shape = (1, 32, 32, 32, 32)
    n_classes = 1
    x = 10 * np.random.random(input_shape)
    y = 10 * np.random.random(input_shape)
    model = unet_lstm(input_shape=(32, 32, 32, 32, 1), n_classes=1)
    actual_output = model.predict(x)
    assert actual_output.shape == y.shape[:-1] + (n_classes,)


def test_vox2vox():
    input_shape = (1, 32, 32, 32, 1)
    n_classes = 1
    x = 10 * np.random.random(input_shape)
    y = np.random.choice([True, False], input_shape)

    # testing ensembler
    model_test(Vox_ensembler, n_classes, input_shape)

    # testing Vox2VoxGan
    vox_generator, vox_discriminator = vox_gan(n_classes, input_shape[1:])

    # testing generator
    vox_generator.compile(tf.optimizers.Adam(), "binary_crossentropy")
    vox_generator.fit(x, y)
    actual_output = vox_generator.predict(x)
    assert actual_output.shape == x.shape[:-1] + (n_classes,)

    # testing descriminator
    pred_shape = (1, 2, 2, 2, 1)
    out = vox_discriminator(inputs=[y, x])
    assert out.shape == pred_shape
