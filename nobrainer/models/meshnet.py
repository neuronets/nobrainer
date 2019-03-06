"""Model definition for MeshNet.

Implemented according to the [MeshNet manuscript](https://arxiv.org/abs/1612.00940)
"""

import tensorflow as tf
from tensorflow.keras import layers

from nobrainer.layers.variational_convolution import VWNConv3D


def meshnet(n_classes, input_shape, receptive_field=67, filters=71, activation='relu', dropout_rate=0.25, batch_size=None, name='meshnet'):
    """Instantiate MeshNet model.

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    receptive_field: {37, 67, 129}, the receptive field of the model. According
        to the MeshNet manuscript, the receptive field should be similar to your
        input shape. The actual receptive field is the cube of the value provided.
    filters: int, number of filters per volumetric convolution. The original
        MeshNet manuscript uses 21 filters for a binary segmentation task
        (i.e., brain extraction) and 71 filters for a multi-class segmentation task.
    activation: str or optimizer object, the non-linearity to use.
    dropout_rate: float between 0 and 1, the fraction of input units to drop.
    batch_size: int, number of samples in each batch. This must be set when
        training on TPUs.
    name: str, name to give to the resulting model object.

    Returns
    -------
    Model object.

    Raises
    ------
    ValueError if receptive field is not an allowable value.
    """

    if receptive_field not in {37, 67, 129}:
        raise ValueError("unknown receptive field. Legal values are 37, 67, and 129.")

    def one_layer(x, layer_num, dilation_rate=(1, 1, 1)):
        x = layers.Conv3D(filters, kernel_size=(3, 3, 3), padding='same', dilation_rate=dilation_rate, name='layer{}/conv3d'.format(layer_num))(x)
        x = layers.BatchNormalization(name='layer{}/batchnorm'.format(layer_num))(x)
        x = layers.Activation(activation, name='layer{}/activation'.format(layer_num))(x)
        x = layers.Dropout(dropout_rate, name='layer{}/dropout'.format(layer_num))(x)
        return x

    inputs = layers.Input(shape=input_shape, batch_size=batch_size, name='inputs')

    if receptive_field == 37:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3)
        x = one_layer(x, 4, dilation_rate=(2, 2, 2))
        x = one_layer(x, 5, dilation_rate=(4, 4, 4))
        x = one_layer(x, 6, dilation_rate=(8, 8, 8))
        x = one_layer(x, 7)
    elif receptive_field == 67:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3, dilation_rate=(2, 2, 2))
        x = one_layer(x, 4, dilation_rate=(4, 4, 4))
        x = one_layer(x, 5, dilation_rate=(8, 8, 8))
        x = one_layer(x, 6, dilation_rate=(16, 16, 16))
        x = one_layer(x, 7)
    elif receptive_field == 129:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2, dilation_rate=(2, 2, 2))
        x = one_layer(x, 3, dilation_rate=(4, 4, 4))
        x = one_layer(x, 4, dilation_rate=(8, 8, 8))
        x = one_layer(x, 5, dilation_rate=(16, 16, 16))
        x = one_layer(x, 6, dilation_rate=(32, 32, 32))
        x = one_layer(x, 7)

    x = layers.Conv3D(filters=n_classes, kernel_size=(1, 1, 1), padding='same', name='classification/conv3d')(x)

    final_activation = 'sigmoid' if n_classes == 1 else 'softmax'
    x = layers.Activation(final_activation, name='classification/activation')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def meshnet_vwn(n_classes, input_shape, receptive_field=67, filters=71, activation='relu', is_mc=False, batch_size=None, name='meshnet_vwn'):
    """Instantiate variational MeshNet model, which also uses weight
    normalization.

    Please see https://arxiv.org/abs/1805.10863 for more information.


    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    receptive_field: {37, 67, 129}, the receptive field of the model. According
        to the MeshNet manuscript, the receptive field should be similar to your
        input shape. The actual receptive field is the cube of the value provided.
    filters: int, number of filters per volumetric convolution. The original
        MeshNet manuscript uses 21 filters for a binary segmentation task
        (i.e., brain extraction) and 71 filters for a multi-class segmentation task.
    activation: str or optimizer object, the non-linearity to use.
    is_mc: bool, if true, apply variational approximation in the variational
        convolution layers. If false, will use the most likely weight (i.e.,
        the mean of the weight gaussian) instead of sampling a weight from the
        gaussian distribution.
    batch_size: int, number of samples in each batch. This must be set when
        training on TPUs.
    name: str, name to give to the resulting model object.

    Returns
    -------
    Model object.

    Raises
    ------
    ValueError if receptive field is not an allowable value.
    """

    if receptive_field not in {37, 67, 129}:
        raise ValueError("unknown receptive field. Legal values are 37, 67, and 129.")

    def one_layer(x, layer_num, dilation_rate=(1, 1, 1)):
        # Normally, meshnet has layers of
        # conv -> batchnorm --> activation --> dropout
        # but for the variational model, we do not include batchnorm and
        # dropout. The variational convolution implements weightnorm, which
        # gives similar benefits as batchnorm without the extra trainable
        # parameters.
        x = VWNConv3D(filters, kernel_size=(3, 3, 3), padding='same', dilation_rate=dilation_rate, is_mc=is_mc, name='layer{}/vwnconv3d'.format(layer_num))(x)
        x = layers.Activation(activation, name='layer{}/activation'.format(layer_num))(x)
        return x

    inputs = layers.Input(shape=input_shape, batch_size=batch_size, name='inputs')

    if receptive_field == 37:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3)
        x = one_layer(x, 4, dilation_rate=(2, 2, 2))
        x = one_layer(x, 5, dilation_rate=(4, 4, 4))
        x = one_layer(x, 6, dilation_rate=(8, 8, 8))
        x = one_layer(x, 7)
    elif receptive_field == 67:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3, dilation_rate=(2, 2, 2))
        x = one_layer(x, 4, dilation_rate=(4, 4, 4))
        x = one_layer(x, 5, dilation_rate=(8, 8, 8))
        x = one_layer(x, 6, dilation_rate=(16, 16, 16))
        x = one_layer(x, 7)
    elif receptive_field == 129:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2, dilation_rate=(2, 2, 2))
        x = one_layer(x, 3, dilation_rate=(4, 4, 4))
        x = one_layer(x, 4, dilation_rate=(8, 8, 8))
        x = one_layer(x, 5, dilation_rate=(16, 16, 16))
        x = one_layer(x, 6, dilation_rate=(32, 32, 32))
        x = one_layer(x, 7)

    x = VWNConv3D(filters=n_classes, kernel_size=(1, 1, 1), padding='same', is_mc=is_mc, name='classification/vwnconv3d')(x)

    final_activation = 'sigmoid' if n_classes == 1 else 'softmax'
    x = layers.Activation(final_activation, name='classification/activation')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
