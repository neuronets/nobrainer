import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from ..layers.InstanceNorm import InstanceNormalization


def vox_gan(
    n_classes,
    input_shape,
    g_filters=64,
    g_kernel_size=4,
    g_norm="batch",
    d_filters=64,
    d_kernel_size=4,
    d_norm="batch",
):
    """Instantiate Vox2VoxGAN.

    Adapted from https://arxiv.org/abs/2003.13653
    Code: https://github.com/mdciri/Vox2Vox

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    g_kernal_size: int, size of the kernel for generator. Default kernel size
        is set to be 4.
    g_filters: int, number of filters for generator. default is set 64.
    g_norm: str, to set batch or instance norm.
    d_kernal_size: int, size of the kernel for discriminator. Default kernel size
        is set to be 4.
    d_filters: int, number of filters for discriminator. default is set 64.
    d_norm: str, to set batch or instance norm.

    Returns
    ----------
    Model object.

    """
    generator = Vox_generator(
        n_classes,
        input_shape,
        n_filters=g_filters,
        kernel_size=g_kernel_size,
        norm=g_norm,
    )

    discriminator = Vox_discriminator(
        input_shape, n_filters=d_filters, kernel_size=d_kernel_size, norm=d_norm
    )
    return generator, discriminator


def Vox_generator(n_classes, input_shape, n_filters=64, kernel_size=4, norm="batch"):
    """Instantiate Generator.

    Adapted from https://arxiv.org/abs/2003.13653

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 4.
    n_filters: int, number of filters. default is set 64.
    norm: str, to set batch or instance norm.

    Returns
    ----------
    Model object.

    """

    def encoder_step(inputs, filters, kernel_size=3, norm="instance"):
        x = layers.Conv3D(
            filters,
            kernel_size=kernel_size,
            strides=2,
            kernel_initializer="he_normal",
            padding="same",
        )(inputs)
        if norm == "instance":
            x = InstanceNormalization()(x)
        if norm == "batch":
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)

        return x

    def bottleneck(inputs, filters, kernel_size, norm="instance"):
        x = layers.Conv3D(
            filters,
            kernel_size=kernel_size,
            strides=2,
            kernel_initializer="he_normal",
            padding="same",
        )(inputs)
        if norm == "instance":
            x = InstanceNormalization()(x)
        if norm == "batch":
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        for i in range(4):
            y = layers.Conv3D(
                filters,
                kernel_size=kernel_size,
                strides=1,
                kernel_initializer="he_normal",
                padding="same",
            )(x)
            if norm == "instance":
                x = InstanceNormalization()(y)
            if norm == "batch":
                x = layers.BatchNormalization()(y)
            x = layers.LeakyReLU()(x)
            x = layers.Concatenate()([x, y])

        return x

    def decoder_step(
        inputs, layer_to_concatenate, filters, kernel_size, norm="instance"
    ):
        x = layers.Conv3DTranspose(
            filters,
            kernel_size,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        if norm == "instance":
            x = InstanceNormalization()(x)
        if norm == "batch":
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Concatenate()([x, layer_to_concatenate])
        x = layers.Dropout(0.2)(x)
        return x

    layers_to_concatenate = []
    inputs = layers.Input(input_shape, name="input_image")
    Nfilter_start = n_filters
    depth = 4
    x = inputs

    # encoder
    for d in range(depth - 1):
        if d == 0:
            x = encoder_step(
                x, Nfilter_start * np.power(2, d), kernel_size, norm="None"
            )
        else:
            x = encoder_step(x, Nfilter_start * np.power(2, d), kernel_size, norm=norm)
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottleneck(x, Nfilter_start * np.power(2, depth - 1), kernel_size, norm=norm)

    # decoder
    for d in range(depth - 2, -1, -1):
        x = decoder_step(
            x,
            layers_to_concatenate.pop(),
            Nfilter_start * np.power(2, d),
            kernel_size,
            norm=norm,
        )

    # classifier
    last = layers.Conv3DTranspose(
        n_classes,
        kernel_size=kernel_size,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        activation="softmax",
        name="output_generator",
    )(x)

    return Model(inputs=inputs, outputs=last, name="Generator")


def Vox_discriminator(input_shape, n_filters=64, kernel_size=4, norm="batch"):
    """Instantiate Discriminator.

    Adapted from https://arxiv.org/abs/2003.13653

    Parameters
    ----------
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    n_filters: int, number of filters. default is set 64.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 4.
    norm: str, to set batch or instance norm.

    Returns
    ----------
    Model object.

    """

    inputs = layers.Input(input_shape, name="input_image")
    targets = layers.Input(input_shape, name="target_image")
    Nfilter_start = n_filters
    depth = 3

    def encoder_step(inputs, n_filters, kernel_size, norm="instance"):
        x = layers.Conv3D(
            n_filters,
            kernel_size,
            strides=2,
            kernel_initializer="he_normal",
            padding="same",
        )(inputs)
        if norm == "instance":
            x = InstanceNormalization()(x)
        if norm == "batch":
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.2)(x)

        return x

    x = layers.Concatenate()([inputs, targets])

    for d in range(depth):
        if d == 0:
            x = encoder_step(
                x, Nfilter_start * np.power(2, d), kernel_size, norm="None"
            )
        else:
            x = encoder_step(x, Nfilter_start * np.power(2, d), kernel_size, norm=norm)

    x = layers.ZeroPadding3D()(x)
    x = layers.Conv3D(
        Nfilter_start * (2**depth),
        kernel_size,
        strides=1,
        padding="valid",
        kernel_initializer="he_normal",
    )(x)
    if norm == "instance":
        x = InstanceNormalization()(x)
    if norm == "batch":
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.ZeroPadding3D()(x)
    last = layers.Conv3D(
        1,
        kernel_size,
        strides=1,
        padding="valid",
        kernel_initializer="he_normal",
        name="output_discriminator",
    )(x)

    return Model(inputs=[targets, inputs], outputs=last, name="Discriminator")


def Vox_ensembler(n_classes, input_shape, kernel_size=3, **kwargs):
    """Instantiate Ensembler.

    Adapted from https://arxiv.org/abs/2003.13653

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.

    Returns
    ----------
    Model object.
    """
    start = layers.Input(input_shape)
    fin = layers.Conv3D(
        n_classes,
        kernel_size=kernel_size,
        kernel_initializer="he_normal",
        padding="same",
        activation="softmax",
    )(start)

    return Model(inputs=start, outputs=fin, name="Ensembler")
