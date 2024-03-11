"""Attention U-net with inception layers.
Adapted from https://github.com/robinvvinod/unet
"""

from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

K.set_image_data_format("channels_last")


def expend_as(tensor, rep):
    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = layers.Lambda(
        lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
        arguments={"repnum": rep},
    )(tensor)
    return my_repeat


def conv3d_block(
    input_tensor,
    n_filters,
    kernel_size=3,
    batchnorm=True,
    strides=1,
    dilation_rate=1,
    recurrent=1,
):
    # A wrapper of the Keras Conv3D block to serve as a building block for downsampling layers
    # Includes options to use batch normalization, dilation and recurrence

    conv = layers.Conv3D(
        filters=n_filters,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer="he_normal",
        padding="same",
        dilation_rate=dilation_rate,
    )(input_tensor)
    if batchnorm:
        conv = layers.BatchNormalization()(conv)
    output = layers.LeakyReLU(alpha=0.1)(conv)

    for _ in range(recurrent - 1):
        conv = layers.Conv3D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            dilation_rate=dilation_rate,
        )(output)
        if batchnorm:
            conv = layers.BatchNormalization()(conv)
        res = layers.LeakyReLU(alpha=0.1)(conv)
        output = layers.Add()([output, res])

    return output


def AttnGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv3D(
        filters=inter_shape, kernel_size=1, strides=1, padding="same"
    )(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv3D(
        filters=inter_shape,
        kernel_size=3,
        strides=(
            shape_x[1] // shape_g[1],
            shape_x[2] // shape_g[2],
            shape_x[3] // shape_g[3],
        ),
        padding="same",
    )(x)

    # Element-wise addition of the gating and x signals
    add_xg = layers.add([phi_g, theta_x])
    add_xg = layers.Activation("relu")(add_xg)

    # 1x1x1 convolution
    psi = layers.Conv3D(filters=1, kernel_size=1, padding="same")(add_xg)
    psi = layers.Activation("sigmoid")(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = layers.UpSampling3D(
        size=(
            shape_x[1] // shape_sigmoid[1],
            shape_x[2] // shape_sigmoid[2],
            shape_x[3] // shape_sigmoid[3],
        )
    )(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[4])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = layers.multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = layers.Conv3D(
        filters=shape_x[4], kernel_size=1, strides=1, padding="same"
    )(attn_coefficients)
    output = layers.BatchNormalization()(output)
    return output


def transpose_block(
    input_tensor,
    skip_tensor,
    n_filters,
    kernel_size=3,
    strides=1,
    batchnorm=True,
    recurrent=1,
):
    # A wrapper of the Keras Conv3DTranspose block to serve as a building block for upsampling layers

    shape_x = K.int_shape(input_tensor)
    shape_xskip = K.int_shape(skip_tensor)

    conv = layers.Conv3DTranspose(
        filters=n_filters,
        kernel_size=kernel_size,
        padding="same",
        strides=(
            shape_xskip[1] // shape_x[1],
            shape_xskip[2] // shape_x[2],
            shape_xskip[3] // shape_x[3],
        ),
        kernel_initializer="he_normal",
    )(input_tensor)
    conv = layers.LeakyReLU(alpha=0.1)(conv)

    act = conv3d_block(
        conv,
        n_filters=n_filters,
        kernel_size=kernel_size,
        strides=1,
        batchnorm=batchnorm,
        dilation_rate=1,
        recurrent=recurrent,
    )
    output = layers.Concatenate(axis=4)([act, skip_tensor])
    return output


# Use the functions provided in layers3D to build the network
def inception_block(
    input_tensor,
    n_filters,
    kernel_size=3,
    strides=1,
    batchnorm=True,
    recurrent=1,
    layers_list=[],
):
    # Inception-style convolutional block similar to InceptionNet
    # The first convolution follows the function arguments, while subsequent inception convolutions follow the parameters in
    # argument, layers

    # layers is a nested list containing the different secondary inceptions in the format of (kernel_size, dil_rate)

    # E.g => layers=[ [(3,1),(3,1)], [(5,1)], [(3,1),(3,2)] ]
    # This will implement 3 sets of secondary convolutions
    # Set 1 => 3x3 dil = 1 followed by another 3x3 dil = 1
    # Set 2 => 5x5 dil = 1
    # Set 3 => 3x3 dil = 1 followed by 3x3 dil = 2

    res = conv3d_block(
        input_tensor,
        n_filters=n_filters,
        kernel_size=kernel_size,
        strides=strides,
        batchnorm=batchnorm,
        dilation_rate=1,
        recurrent=recurrent,
    )

    temp = []
    for layer in layers_list:
        local_res = res
        for conv in layer:
            incep_kernel_size = conv[0]
            incep_dilation_rate = conv[1]
            local_res = conv3d_block(
                local_res,
                n_filters=n_filters,
                kernel_size=incep_kernel_size,
                strides=1,
                batchnorm=batchnorm,
                dilation_rate=incep_dilation_rate,
                recurrent=recurrent,
            )
        temp.append(local_res)

    temp = layers.concatenate(temp)
    res = conv3d_block(
        temp,
        n_filters=n_filters,
        kernel_size=1,
        strides=1,
        batchnorm=batchnorm,
        dilation_rate=1,
    )

    shortcut = conv3d_block(
        input_tensor,
        n_filters=n_filters,
        kernel_size=1,
        strides=strides,
        batchnorm=batchnorm,
        dilation_rate=1,
    )
    if batchnorm:
        shortcut = layers.BatchNormalization()(shortcut)

    output = layers.Add()([shortcut, res])
    return output


def attention_unet_with_inception(
    n_classes, input_shape, batch_size=None, n_filters=16, batchnorm=True
):
    # contracting path

    inputs = layers.Input(shape=input_shape, batch_size=batch_size)

    c0 = inception_block(
        inputs,
        n_filters=n_filters,
        batchnorm=batchnorm,
        strides=1,
        recurrent=2,
        layers_list=[[(3, 1), (3, 1)], [(3, 2)]],
    )  # 512x512x512

    c1 = inception_block(
        c0,
        n_filters=n_filters * 2,
        batchnorm=batchnorm,
        strides=2,
        recurrent=2,
        layers_list=[[(3, 1), (3, 1)], [(3, 2)]],
    )  # 256x256x256

    c2 = inception_block(
        c1,
        n_filters=n_filters * 4,
        batchnorm=batchnorm,
        strides=2,
        recurrent=2,
        layers_list=[[(3, 1), (3, 1)], [(3, 2)]],
    )  # 128x128x128

    c3 = inception_block(
        c2,
        n_filters=n_filters * 8,
        batchnorm=batchnorm,
        strides=2,
        recurrent=2,
        layers_list=[[(3, 1), (3, 1)], [(3, 2)]],
    )  # 64x64x64

    # bridge

    b0 = inception_block(
        c3,
        n_filters=n_filters * 16,
        batchnorm=batchnorm,
        strides=2,
        recurrent=2,
        layers_list=[[(3, 1), (3, 1)], [(3, 2)]],
    )  # 32x32x32

    # expansive path

    attn0 = AttnGatingBlock(c3, b0, n_filters * 16)
    u0 = transpose_block(
        b0, attn0, n_filters=n_filters * 8, batchnorm=batchnorm, recurrent=2
    )  # 64x64x64

    attn1 = AttnGatingBlock(c2, u0, n_filters * 8)
    u1 = transpose_block(
        u0, attn1, n_filters=n_filters * 4, batchnorm=batchnorm, recurrent=2
    )  # 128x128x128

    attn2 = AttnGatingBlock(c1, u1, n_filters * 4)
    u2 = transpose_block(
        u1, attn2, n_filters=n_filters * 2, batchnorm=batchnorm, recurrent=2
    )  # 256x256x256

    u3 = transpose_block(
        u2, c0, n_filters=n_filters, batchnorm=batchnorm, recurrent=2
    )  # 512x512x512

    outputs = layers.Conv3D(filters=1, kernel_size=1, strides=1)(u3)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    outputs = layers.Activation(final_activation)(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


if __name__ == "__main__":
    model = attention_unet_with_inception(n_classes=1, input_shape=(256, 256, 256, 1))
    model.summary()
