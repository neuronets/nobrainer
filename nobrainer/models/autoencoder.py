"""Model definition for Autoencoder.
"""
import math

from tensorflow.keras import layers, models


def autoencoder(
    input_shape,
    encoding_dim=512,
    n_base_filters=16,
    batchnorm=True,
    batch_size=None,
):
    """Instantiate Autoencoder Architecture.

    Parameters
    ----------
    input_shape: list or tuple of four ints, the shape of the input data. Should be
        scaled to [0,1]. Omit the batch dimension, and include the number of channels.
        Currently, only squares and cubes supported.
    encoding_dim: int, the dimensions of the encoding of the input data. This would
        translate to a latent code of dimensions encoding_dimx1.
    n_base_filters: int, number of base filters the models first convolutional layer.
        The subsequent layers have n_filters which are multiples of n_base_filters.
    batchnorm: bool, whether to use batch normalization in the network.
    batch_size: int, number of samples in each batch. This must be set when training on
        TPUs.
    name: str, name to give to the resulting model object.

    Returns
    -------
    Model object.
    """

    conv_kwds = {"kernel_size": 4, "activation": None, "padding": "same", "strides": 2}

    conv_transpose_kwds = {
        "kernel_size": 4,
        "strides": 2,
        "activation": None,
        "padding": "same",
    }

    dimensions = input_shape[:-1]
    n_dims = len(dimensions)

    if not (n_dims in [2, 3] and dimensions[1:] == dimensions[:-1]):
        raise ValueError("Dimensions should be of square or cube!")

    Conv = getattr(layers, "Conv{}D".format(n_dims))
    ConvTranspose = getattr(layers, "Conv{}DTranspose".format(n_dims))
    n_layers = int(math.log(dimensions[0], 2))

    # Input layer
    inputs = x = layers.Input(shape=input_shape, batch_size=batch_size, name="inputs")

    # Encoder
    for i in range(n_layers):
        n_filters = min(n_base_filters * (2 ** (i)), encoding_dim)

        x = Conv(n_filters, **conv_kwds)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

    # Encoding of the input image
    x = layers.Flatten(name="Encoding")(x)

    # Decoder
    x = layers.Reshape((1,) * n_dims + (encoding_dim,))(x)
    for i in range(n_layers)[::-1]:
        n_filters = min(n_base_filters * (2 ** (i)), encoding_dim)

        x = ConvTranspose(n_filters, **conv_transpose_kwds)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

    # Output layer
    outputs = Conv(1, 3, activation="sigmoid", padding="same")(x)

    return models.Model(inputs=inputs, outputs=outputs)
