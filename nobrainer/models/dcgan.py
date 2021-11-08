"""Model definition for DCGAN.
"""
import math

from tensorflow.keras import layers, models


def dcgan(
    output_shape,
    z_dim=256,
    n_base_filters=16,
    batchnorm=True,
    batch_size=None,
    name="dcgan",
):
    """Instantiate DCGAN Architecture.

    Parameters
    ----------
    output_shape: list or tuple of four ints, the shape of the output images. Should be
        scaled to [0,1]. Omit the batch dimension, and include the number of channels.
        Currently, only squares and cubes supported.
    z_dim: int, the dimensions of the encoding of the latent code. This would translate
        to a latent code of dimensions encoding_dimx1.
    n_base_filters: int, number of base filters the models first convolutional layer.
        The subsequent layers have n_filters which are multiples of n_base_filters.
    batchnorm: bool, whether to use batch normalization in the network.
    batch_size: int, number of samples in each batch. This must be set when
        training on TPUs.
    name: str, name to give to the resulting model object.

    Returns
    -------
    Generator Model object.
    Discriminator Model object.
    """

    conv_kwds = {"kernel_size": 4, "activation": None, "padding": "same", "strides": 2}

    conv_transpose_kwds = {
        "kernel_size": 4,
        "strides": 2,
        "activation": None,
        "padding": "same",
    }

    dimensions = output_shape[:-1]
    n_dims = len(dimensions)

    if not (n_dims in [2, 3] and dimensions[1:] == dimensions[:-1]):
        raise ValueError("Dimensions should be of square or cube!")

    Conv = getattr(layers, "Conv{}D".format(n_dims))
    ConvTranspose = getattr(layers, "Conv{}DTranspose".format(n_dims))
    n_layers = int(math.log(dimensions[0], 2))

    # Generator
    z_input = layers.Input(shape=(z_dim,), batch_size=batch_size)

    project = layers.Dense(pow(4, n_dims) * z_dim)(z_input)
    project = layers.ReLU()(project)
    project = layers.Reshape((4,) * n_dims + (z_dim,))(project)
    x = project

    for i in range(n_layers - 2)[::-1]:
        n_filters = min(n_base_filters * (2 ** (i)), z_dim)

        x = ConvTranspose(n_filters, **conv_transpose_kwds)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    outputs = Conv(1, 3, activation="sigmoid", padding="same")(x)

    generator = models.Model(
        inputs=[z_input], outputs=[outputs], name=name + "_generator"
    )

    # PatchGAN Discriminator with output of 8x8(x8)
    inputs = layers.Input(shape=(output_shape), batch_size=batch_size)
    x = inputs
    for i in range(n_layers - 3):
        n_filters = min(n_base_filters * (2 ** (i)), z_dim)

        x = Conv(n_filters, **conv_kwds)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    pred = Conv(1, 3, padding="same", activation="sigmoid")(x)

    discriminator = models.Model(
        inputs=[inputs], outputs=[pred], name=name + "_discriminator"
    )

    return generator, discriminator
