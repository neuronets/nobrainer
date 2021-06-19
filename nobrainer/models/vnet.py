# Adaptation of the Vnet model from https://arxiv.org/pdf/1606.04797.pdf with dropouts and

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
    convd = Conv3D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    convd = GroupNormalization()(convd)
    convd = Conv3D(filters, kernel_size, activation=activation, padding=padding)(convd)
    convd = GroupNormalization()(convd)
    pool = MaxPooling3D()(convd)
    return convd, pool


def up_stage(inputs, skip, filters, kernel_size=3, activation="relu", padding="SAME"):
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
    input_shape=(256, 256, 256, 1),
    kernel_size=3,
    activation="relu",
    padding="SAME",
    **kwargs
):
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
