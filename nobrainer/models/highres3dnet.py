"""HighRes3DNet implemented in TensorFlow.

Reference
---------
Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
On the Compactness, Efficiency, and Representation of 3D Convolutional
Networks: Brain Parcellation as a Pretext Task. In: Niethammer M. et al. (eds)
Information Processing in Medical Imaging. IPMI 2017. Lecture Notes in Computer
Science, vol 10265.
"""

import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

from nobrainer.models import util

FUSED_BATCHED_NORM = True

# TODO: use `tensorflow.python.framework.test_util.is_gpu_available` to
# determine data format. `channels_first` is typically faster on GPU, and
# `channels_last` is typically faster on CPU.
# TODO: Once the above todo is implemented, add `channel_format` arguments.


def _resblock(inputs, layer_num, mode, filters, kernel_size=3, dilation_rate=1,
              save_activations=False):
    """Return building block of residual network. This includes the residual
    connection.

    See the notes below for an overview of the operations performed in this
    function.

    Parameters
    ----------
    inputs : numeric Tensor
        Input Tensor.
    layer_num : int
        Value to append to each operator name. This should be the layer number
        in the network.
    mode : string
        A TensorFlow mode key.
    filters : int or tuple
        Number of 3D convolution filters.
    kernel_size : int or tuple
        Size of 3D convolution kernel.
    dilation_rate : int or tuple
        Rate of dilution in 3D convolution.
    save_activations : boolean
        If true, save activations to histogram.

    Returns
    -------
    Numeric Tensor of same type as `inputs`.

    Notes
    -----
        +-inputs-+
        |        |
        |    batchnorm
        |        |
        |      relu
        |        |
        |     conv3d
        |        |
        |    batchnorm
        |        |
        |      relu
        |        |
        |     conv3d
        |        |
        +-(sum)--+
            |
         outputs
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('batchnorm_{}_0'.format(layer_num)):
        bn1 = tf.layers.batch_normalization(
            inputs, training=training, fused=FUSED_BATCHED_NORM,
        )
    with tf.variable_scope('relu_{}_0'.format(layer_num)):
        relu1 = tf.nn.relu(bn1)
    with tf.variable_scope('conv_{}_0'.format(layer_num)):
        conv1 = tf.layers.conv3d(
            relu1, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )
        if save_activations:
            util._add_activation_summary(conv1)

    with tf.variable_scope('batchnorm_{}_1'.format(layer_num)):
        bn2 = tf.layers.batch_normalization(
            conv1, training=training, fused=FUSED_BATCHED_NORM,
        )
    with tf.variable_scope('relu_{}_1'.format(layer_num)):
        relu2 = tf.nn.relu(bn2)
    with tf.variable_scope('conv_{}_1'.format(layer_num)):
        conv2 = tf.layers.conv3d(
            relu2, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )
        if save_activations:
            util._add_activation_summary(conv2)

    with tf.variable_scope('add_{}'.format(layer_num)):
        return tf.add(conv2, inputs)


def _highres3dnet_logit_fn(features, num_classes, mode,
                           save_activations=False):
    """HighRes3DNet logit function.

    Parameters
    ----------
    features : numeric Tensor
        Input Tensor.
    num_classes : int
        Number of classes to segment. This is the number of filters in the
        final convolution layer.
    mode : string
        A TensorFlow mode key.
    save_activations : boolean
        If true, save activations to histogram.

    Returns
    -------
    Tensor of logits.
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('conv_0'):
        conv = tf.layers.conv3d(
            features, filters=16, kernel_size=3, padding='SAME',
        )
    with tf.variable_scope('batchnorm_0'):
        conv = tf.layers.batch_normalization(
            conv, training=training, fused=FUSED_BATCHED_NORM,
        )
    with tf.variable_scope('relu_0'):
        outputs = tf.nn.relu(conv)

    if save_activations:
        util._add_activation_summary(outputs)

    for ii in range(3):
        offset = 1
        layer_num = ii + offset
        outputs = _resblock(
            outputs, layer_num=layer_num, mode=mode, filters=16,
        )

    for ii in range(3):
        offset = 4
        layer_num = ii + offset
        outputs = _resblock(
            outputs, layer_num=layer_num, mode=mode, filters=16,
            dilation_rate=2,
        )

    for ii in range(3):
        offset = 7
        layer_num = ii + offset
        outputs = _resblock(
            outputs, layer_num=layer_num, mode=mode, filters=16,
            dilation_rate=4,
        )

    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            outputs, filters=num_classes, kernel_size=1, padding='SAME',
        )
        if save_activations:
            util._add_activation_summary(logits)

    return logits


def _highres3dnet_model_fn(features, labels, mode, num_classes,
                           optimizer='Adam', learning_rate=0.001, config=None):
    """"""
    logits = _highres3dnet_logit_fn(
        features=features, mode=mode, num_classes=num_classes,
    )
    predictions = tf.argmax(logits, axis=-1)

    optimizer_ = optimizers.get_optimizer_instance(
        optimizer, learning_rate=learning_rate
    )

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits,
    )
    # cross_entropy.shape == (batch_size, *block_shape)
    loss = tf.reduce_mean(cross_entropy)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer_.minimize(
            loss, global_step=tf.train.get_global_step(),
        )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
    )


class HighRes3DNet(tf.estimator.Estimator):
    """"""
    def __init__(self, num_classes, model_dir=None, optimizer='Adam',
                 learning_rate=0.001, warm_start_from=None, config=None):

        def _model_fn(features, labels, mode, config):
            """"""
            return _highres3dnet_model_fn(
                features=features, labels=labels, mode=mode,
                num_classes=num_classes,
                optimizer=optimizer, learning_rate=learning_rate,
                config=config,
            )

        super(HighRes3DNet, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from,
        )
