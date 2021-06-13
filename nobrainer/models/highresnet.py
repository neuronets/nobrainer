"""Model definition for HighResNet.

Implemented according to the [HighResNet manuscript](https://arxiv.org/abs/1707.01992).
"""

import tensorflow as tf
from tensorflow.keras import layers

from ..layers.padding import ZeroPadding3DChannels


def highresnet(
    n_classes, input_shape, activation="relu", dropout_rate=0, name="highresnet"
):
    """Instantiate HighResNet model."""

    conv_kwds = {"kernel_size": (3, 3, 3), "padding": "same"}

    n_base_filters = 16

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(n_base_filters, **conv_kwds)(inputs)

    for ii in range(3):
        skip = x
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv3D(n_base_filters, **conv_kwds)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv3D(n_base_filters, **conv_kwds)(x)
        x = layers.Add()([x, skip])

    x = ZeroPadding3DChannels(8)(x)
    for ii in range(3):
        skip = x
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv3D(n_base_filters * 2, dilation_rate=2, **conv_kwds)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv3D(n_base_filters * 2, dilation_rate=2, **conv_kwds)(x)
        x = layers.Add()([x, skip])

    x = ZeroPadding3DChannels(16)(x)
    for ii in range(3):
        skip = x
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv3D(n_base_filters * 4, dilation_rate=4, **conv_kwds)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv3D(n_base_filters * 4, dilation_rate=4, **conv_kwds)(x)
        x = layers.Add()([x, skip])

    x = layers.Conv3D(filters=n_classes, kernel_size=(1, 1, 1), padding="same")(x)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    x = layers.Activation(final_activation)(x)

    # QUESTION: where should dropout go?

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
