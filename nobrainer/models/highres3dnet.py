"""HighRes3DNet implemented in TensorFlow.

Notes
-----
This script was written with the help of
https://github.com/tensorflow/models/blob/178480ed3be30bc33dafead53b5ee09c717fa2b7/official/resnet/resnet_model.py.

Reference
---------
Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
On the Compactness, Efficiency, and Representation of 3D Convolutional
Networks: Brain Parcellation as a Pretext Task. In: Niethammer M. et al. (eds)
Information Processing in Medical Imaging. IPMI 2017. Lecture Notes in Computer
Science, vol 10265.
"""

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

tf.logging.set_verbosity(tf.logging.INFO)


def _batch_norm_relu(inputs, is_training, data_format):
    """Perform batch normalization and ReLU on `inputs`."""
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True,
    )
    return tf.nn.relu(inputs)


def _building_block(inputs, filters, dilation_rate, is_training, strides,
                    data_format):
    """Perform two rounds of batch normalization, ReLU, and 3D convolution,
    and sum original inputs with ouputs."""
    shortcut = inputs

    for _ in range(2):
        inputs = _batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.conv3d(
            inputs=inputs, filters=filters, kernel_size=(3, 3, 3),
            strides=strides, padding='same', data_format=data_format,
            dilation_rate=dilation_rate,
        )

    return inputs + shortcut


def _block_layer(inputs, filters, dilation_rate, num_blocks, strides,
                 is_training, name, data_format):
    """Return blocks of blocks buildings."""
    for _ in range(1, num_blocks):
        inputs = _building_block(
            inputs=inputs, filters=filters, dilation_rate=dilation_rate,
            is_training=is_training, strides=strides, data_format=data_format,
        )
    return tf.identity(inputs, name=name)


def highres3dnet(features, labels, num_classes, is_training, data_format=None):
    """Return a HighRes3DNet model.

    Parameters
    ----------
    num_classes : int
        Number of classes to output in the last convolutional layer.
    input_tensor : optional tensor to use as image input for the model. Does
        not have to be provided if `input_shape` is given.
    input_shape : optional shape tuple. Does not have to be provided if
        `input_tensor` is given.

    Returns
    -------
    A TensorFlow model.
    """
    if data_format is None:
        # 'channels_first' is typically faster on GPU. 'channels_last' is
        # typically faster on CPU.
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda()
            else 'channels_last'
        )

    # Initial block.
    inputs = tf.layers.conv3d(
        inputs=features, filters=16, kernel_size=(3, 3, 3), padding='same',
        name='initial_conv',
    )
    inputs = _batch_norm_relu(
        inputs, is_training=is_training, data_format=data_format
    )
    inputs = tf.identity(inputs, name='initial_block')

    # 3 residual blocks.
    inputs = _block_layer(
        inputs=inputs, filters=16, dilation_rate=(1, 1, 1), num_blocks=3,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format,
    )

    # 3 residual blocks.
    inputs = _block_layer(
        inputs=inputs, filters=32, dilation_rate=(2, 2, 2), num_blocks=3,
        strides=1, is_training=is_training, name='block_layer2',
        data_format=data_format,
    )

    # 3 residual blocks.
    inputs = _block_layer(
        inputs=inputs, filters=64, dilation_rate=(4, 4, 4), num_blocks=3,
        strides=1, is_training=is_training, name='block_layer3',
        data_format=data_format,
    )

    # Final classification layer.
    inputs = tf.layers.conv3d(
        inputs=inputs, filters=num_classes, kernel_size=(1, 1, 1),
        padding='same',
    )
    probabilities = tf.nn.softmax(logits=inputs)
    probabilities = tf.identity(inputs, name='classification')

    return probabilities
