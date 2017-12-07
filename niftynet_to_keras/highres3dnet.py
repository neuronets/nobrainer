"""HighRes3DNet for Keras.

Example
-------
>>> volume_shape = (256, 256, 256, 1)
>>> model = HighRes3DNet(n_classes=60, input_shape=volume_shape)
>>> model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])

Reference
---------
[On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task](https://doi.org/10.1007/978-3-319-59050-9_28)
"""

from keras import backend as K
from keras import Model
from keras.layers import (Activation, Add, BatchNormalization, Conv3D, Input)


def dice_coef(y_true, y_pred, smooth=1):
    """Return Dice coefficient of boolean arrays `y_true` and `y_pred`."""
    # https://github.com/fchollet/keras/issues/3611
    axis = (1, 2, 3)  # TODO: investigate which axes are correct.
    intersection = K.sum(y_true * y_pred, axis=axis)
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_loss(y_true, y_pred, smooth=1):
    return 1 - dice_coef(y_true, y_pred, smooth)


def HighRes3DNet(n_classes, input_tensor=None, input_shape=None):
    """Instantiates the HighRes3DNet architecture.

    Parameters
    ----------
    input_tensor : optional Keras tensor (i.e., output of `layers.Input()`) to
        use as image input for the model. Does not have to be provided if
        `input_shape` is given.
    input_shape : optional shape tuple. Does not have to be provided if
        `input_tensor` is given.

    Returns
    -------
        A Keras model instance.
    """
    if input_tensor is not None:
        inputs = input_tensor
    elif input_shape is not None:
        inputs = Input(shape=input_shape, name='input')
    else:
        raise ValueError("Either `input_tensor` or `input_shape` is required.")

    # Block 1
    x = Conv3D(16, (3, 3, 3), padding='same')(inputs)
    x = BatchNormalization(name='norm_0')(x)
    x = Activation('relu')(x)

    # Blocks 2-7 (3 groups with residual connections)
    for _ in range(3):
        residual = x
        for ii in range(2):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv3D(16, (3, 3, 3), padding='same')(x)
        x = Add()([residual, x])

    # Blocks 8-13 (3 groups with residual connections)
    for _ in range(3):
        residual = x
        for ii in range(2):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # TODO: confirm that filters=16 and dilation_rate=(2, 2, 2) gives a
            # convolution with 32 kernels dilated by 2.
            x = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=(2, 2, 2))(x)
        x = Add()([residual, x])

    # Blocks 14-19 (3 groups with residual connections)
    for _ in range(3):
        residual = x
        for ii in range(2):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # TODO: confirm that filters=16 and dilation_rate=(4, 4, 4) gives a
            # convolution with 64 kernels dilated by 4.
            x = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=(4, 4, 4))(x)
        x = Add()([residual, x])

    # Block 20
    x = Conv3D(n_classes, (1, 1, 1))(x)
    x = Activation('softmax')(x)

    return Model(inputs, x, name='highres3dnet')
