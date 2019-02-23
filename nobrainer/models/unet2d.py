"""U-net architecture.

https://arxiv.org/abs/1505.04597
https://github.com/jocicmarko/ultrasound-nerve-segmentation
"""

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.regularizers import l2


def unet2d(input_shape=(224, 224, 1), dropout_rate=0.5):
    """Instantiate two-dimensional U-Net architecture."""

    conv_kwds = {
        'kernel_size': (3, 3),
        'activation': 'relu',
        'padding': 'same',
        'kernel_regularizer': l2(0.1),
    }

    conv_transpose_kwds = {
        'kernel_size': (2, 2),
        'strides': (2, 2),
        'padding': 'same',
        'kernel_regularizer': l2(0.1),
    }

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, **conv_kwds)(inputs)
    conv1 = Conv2D(64, **conv_kwds)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, **conv_kwds)(pool1)
    conv2 = Conv2D(128, **conv_kwds)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, **conv_kwds)(pool2)
    conv3 = Conv2D(256, **conv_kwds)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, **conv_kwds)(pool3)
    conv4 = Conv2D(512, **conv_kwds)(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, **conv_kwds)(pool4)
    conv5 = Conv2D(1024, **conv_kwds)(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    t = Conv2DTranspose(512, **conv_transpose_kwds)(drop5)
    up6 = Concatenate(axis=-1)([t, drop4])
    conv6 = Conv2D(512, **conv_kwds)(up6)
    conv6 = Conv2D(512, **conv_kwds)(conv6)

    t = Conv2DTranspose(256, **conv_transpose_kwds)(conv6)
    up7 = Concatenate(axis=-1)([t, conv3])
    conv7 = Conv2D(256, **conv_kwds)(up7)
    conv7 = Conv2D(256, **conv_kwds)(conv7)

    t = Conv2DTranspose(128, **conv_transpose_kwds)(conv7)
    up8 = Concatenate(axis=-1)([t, conv2])
    conv8 = Conv2D(128, **conv_kwds)(up8)
    conv8 = Conv2D(128, **conv_kwds)(conv8)

    t = Conv2DTranspose(64, **conv_transpose_kwds)(conv8)
    up9 = Concatenate(axis=-1)([t, conv1])
    conv9 = Conv2D(64, **conv_kwds)(up9)
    conv9 = Conv2D(64, **conv_kwds)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=l2(0.1))(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
