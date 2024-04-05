# Adaptation of the VNet model from https://arxiv.org/pdf/1606.04797.pdf
# This 3D deep neural network model is regularized with 3D spatial dropout
# and Group normalization.

from tensorflow.keras.layers import (
    Conv3D,
    Input,
    MaxPooling3D,
    SpatialDropout3D,
    UpSampling3D,
    concatenate,
)
from tensorflow.keras.models import Model

from ..layers.groupnorm import GroupNormalization


def down_stage(inputs, filters, kernel_size=3, activation="relu", padding="SAME"):
    """encoding block of the VNet model.

    Parameters
    ----------
    inputs: tf.layer for encoding stage.
    filters: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Returns
    ----------
    encoding module.
    """
    convd = Conv3D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    convd = GroupNormalization()(convd)
    convd = Conv3D(filters, kernel_size, activation=activation, padding=padding)(convd)
    convd = GroupNormalization()(convd)
    pool = MaxPooling3D()(convd)
    return convd, pool


def up_stage(inputs, skip, filters, kernel_size=3, activation="relu", padding="SAME"):
    """decoding block of the VNet model.

    Parameters
    ----------
    inputs: tf.layer for encoding stage.
    filters: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Returns
    ----------
    decoded module.
    """
    up = UpSampling3D()(inputs)
    up = Conv3D(filters, 2, activation=activation, padding=padding)(up)
    up = GroupNormalization()(up)

    merge = concatenate([skip, up])
    merge = GroupNormalization()(merge)

    convu = Conv3D(filters, kernel_size, activation=activation, padding=padding)(merge)
    convu = GroupNormalization()(convu)
    convu = Conv3D(filters, kernel_size, activation=activation, padding=padding)(convu)
    convu = GroupNormalization()(convu)
    convu = SpatialDropout3D(0.5)(convu, training=True)

    return convu


def end_stage(inputs, n_classes=1, kernel_size=3, activation="relu", padding="SAME"):
    """last logit layer.

    Parameters
    ----------
    inputs: tf.model layer.
    n_classes: int, for binary class use the value 1.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Result
    ----------
    prediction probabilities
    """
    conv = Conv3D(
        filters=n_classes,
        kernel_size=kernel_size,
        activation=activation,
        padding="SAME",
    )(inputs)
    if n_classes == 1:
        conv = Conv3D(n_classes, 1, activation="sigmoid")(conv)
    else:
        conv = Conv3D(n_classes, 1, activation="softmax")(conv)

    return conv


def vnet(
    n_classes=1,
    input_shape=(128, 128, 128, 1),
    kernel_size=3,
    activation="relu",
    padding="SAME",
    **kwargs
):
    """Instantiate a 3D VNet Architecture.

    VNet model: a 3D deep neural network model adapted from
    https://arxiv.org/pdf/1606.04797.pdf adatptations include groupnorm
    and spatial dropout.

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Returns
    ----------
    Model object.

    """
    inputs = Input(input_shape)

    conv1, pool1 = down_stage(
        inputs, 16, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv2, pool2 = down_stage(
        pool1, 32, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv3, pool3 = down_stage(
        pool2, 64, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv4, _ = down_stage(
        pool3, 128, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv4 = SpatialDropout3D(0.5)(conv4, training=True)

    conv5 = up_stage(
        conv4,
        conv3,
        64,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )
    conv6 = up_stage(
        conv5,
        conv2,
        32,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )
    conv7 = up_stage(
        conv6,
        conv1,
        16,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )

    conv8 = end_stage(
        conv7,
        n_classes=n_classes,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )

    return Model(inputs=inputs, outputs=conv8)
