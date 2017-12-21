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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Activation, Add, BatchNormalization, Conv3D, Input
)


# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/27c47a8f38f11e446e33465146c8eb6074872678/train.py#L23-L27
def dice_coef(y_true, y_pred, smooth=1.):
    """Return Dice coefficient given two boolean ndarrays."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    """Return Dice loss given two boolean ndarrays."""
    return 1 - dice_coef(y_true, y_pred, smooth)


def HighRes3DNet(n_classes, input_tensor=None, input_shape=None):
    """Instantiates the HighRes3DNet architecture.
    Parameters
    ----------
    n_classes : int
        Number of classes to output in the last convolutional layer.
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
        raise ValueError("`input_tensor` or `input_shape` must be provided.")

    # Block 0
    x = Conv3D(16, (3, 3, 3), padding='same', name='conv_0')(inputs)
    x = BatchNormalization(name='norm_0')(x)
    x = Activation('relu', name='activ_0')(x)

    names = {
        'activ': 'activ_{}_{}',
        'conv': 'conv_{}_{}',
        'norm': 'norm_{}_{}',
        'add': 'add_{}',
    }

    # Blocks 1-3 (each block has 2 conv3d layers).
    _offset = 1
    for ii in range(3):
        residual = x
        for jj in range(2):
            # suffix = "_{}".format(2 * ii + jj + _offset)
            suffix = "_{}_{}".format(ii + _offset, jj)
            x = BatchNormalization(name='norm' + suffix)(x)
            x = Activation('relu', name='activ' + suffix)(x)
            x = Conv3D(16, (3, 3, 3), padding='same', name='conv' + suffix)(x)
        x = Add(name=names['add'].format(ii + _offset))([residual, x])

    # Blocks 4-6 (each block has 2 conv3d layers).
    for ii in range(3):
        residual = x
        _offset = 4
        for jj in range(2):
            suffix = "_{}_{}".format(ii + _offset, jj)
            x = BatchNormalization(name='norm' + suffix)(x)
            x = Activation('relu', name='activ' + suffix)(x)
            # TODO: confirm that filters=16 and dilation_rate=(2, 2, 2) gives a
            # convolution with 32 kernels dilated by 2.
            x = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=(2, 2, 2),
                       name='conv' + suffix)(x)
        x = Add(name=names['add'].format(ii + _offset))([residual, x])

    # Blocks 7-9 (each block has 2 conv3d layers).
    for ii in range(3):
        residual = x
        _offset = 7
        for jj in range(2):
            suffix = "_{}_{}".format(ii + _offset, jj)
            x = BatchNormalization(name='norm' + suffix)(x)
            x = Activation('relu', name='activ' + suffix)(x)
            # TODO: confirm that filters=16 and dilation_rate=(4, 4, 4) gives a
            # convolution with 64 kernels dilated by 4.
            x = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=(4, 4, 4),
                       name='conv' + suffix)(x)
        x = Add(name=names['add'].format(ii + _offset))([residual, x])

    # Block 19
    x = Conv3D(n_classes, (1, 1, 1), name='classification')(x)
    x = Activation('softmax', name='final_activation')(x)

    return Model(inputs, x, name='highres3dnet')

