"""Model definition for 3D U-Net.

Implemented according to the [3D U-Net manuscript](https://arxiv.org/abs/1606.06650)
"""
import tensorflow as tf
from tensorflow.keras import layers


def unet(
    n_classes,
    input_shape,
    activation="relu",
    batchnorm=False,
    batch_size=None,
    name="unet",
):
    """Instantiate 3D U-Net architecture."""

    conv_kwds = {
        "kernel_size": (3, 3, 3),
        "activation": None,
        "padding": "same",
        # 'kernel_regularizer': tf.keras.regularizers.l2(0.001),
    }

    conv_transpose_kwds = {
        "kernel_size": (2, 2, 2),
        "strides": 2,
        "padding": "same",
        # 'kernel_regularizer': tf.keras.regularizers.l2(0.001),
    }

    n_base_filters = 16
    inputs = layers.Input(shape=input_shape, batch_size=batch_size)

    # Begin analysis path (encoder).

    x = layers.Conv3D(n_base_filters, **conv_kwds)(inputs)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 2, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    skip_1 = x = layers.Activation(activation)(x)
    x = layers.MaxPool3D(2)(x)

    x = layers.Conv3D(n_base_filters * 2, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 4, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    skip_2 = x = layers.Activation(activation)(x)
    x = layers.MaxPool3D(2)(x)

    x = layers.Conv3D(n_base_filters * 4, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 8, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    skip_3 = x = layers.Activation(activation)(x)
    x = layers.MaxPool3D(2)(x)

    x = layers.Conv3D(n_base_filters * 8, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 16, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    # End analysis path (encoder).
    # Begin synthesis path (decoder).

    x = layers.Conv3DTranspose(n_base_filters * 16, **conv_transpose_kwds)(x)

    x = layers.Concatenate(axis=-1)([skip_3, x])
    x = layers.Conv3D(n_base_filters * 8, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 8, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv3DTranspose(n_base_filters * 8, **conv_transpose_kwds)(x)

    x = layers.Concatenate(axis=-1)([skip_2, x])
    x = layers.Conv3D(n_base_filters * 4, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 4, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv3DTranspose(n_base_filters * 4, **conv_transpose_kwds)(x)

    x = layers.Concatenate(axis=-1)([skip_1, x])
    x = layers.Conv3D(n_base_filters * 2, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv3D(n_base_filters * 2, **conv_kwds)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv3D(filters=n_classes, kernel_size=1)(x)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    x = layers.Activation(final_activation)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
