"""MeshNet implemented in TensorFlow.

Reference
---------
Fedorov, A., Johnson, J., Damaraju, E., Ozerin, A., Calhoun, V., & Plis, S.
(2017, May). End-to-end learning of brain tissue segmentation from imperfect
labeling. IJCNN 2017. (pp. 3785-3792). IEEE.
"""

import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

from nobrainer.models import util

FUSED_BATCHED_NORM = True


def _layer(inputs, layer_num, mode, filters, dropout_rate, kernel_size=3,
           dilation_rate=1, save_activations=False):
    """Return layer building block of MeshNet.

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
    dropout_rate : float
        The dropout rate between 0 and 1.
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
    `inputs - conv3d - relu - batchnorm - dropout - outputs`
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('layer_{}'.format(layer_num)):
        conv = tf.layers.conv3d(
            inputs, filters=filters, kernel_size=kernel_size,
            padding='SAME', dilation_rate=dilation_rate,
        )
        relu = tf.nn.relu(conv)
        bn = tf.layers.batch_normalization(
            relu, training=training, fused=FUSED_BATCHED_NORM,
        )
        dropout = tf.layers.dropout(bn, rate=dropout_rate, training=training)

        if save_activations:
            util._add_activation_summary(dropout)

        return dropout


def _meshnet_logit_fn(features, num_classes, mode, dropout_rate=0.25,
                      save_activations=False):
    """MeshNet logit function.

    Parameters
    ----------
    features : numeric Tensor
        Input Tensor.
    num_classes : int
        Number of classes to segment. This is the number of filters in the
        final convolution layer.
    mode : string
        A TensorFlow mode key.
    dropout_rate : float
        The dropout rate between 0 and 1.
    save_activations : boolean
        If true, save activations to histogram.

    Returns
    -------
    Tensor of logits.
    """
    # Dilation rate by layer.
    dilation_rates = (
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (2, 2, 2),
        (4, 4, 4),
        (8, 8, 8),
        (1, 1, 1),
    )

    # All convolution layers use this number of filters.
    filters = 21
    outputs = features

    for ii in range(7):
        dilation_rate = dilation_rates[ii]
        outputs = _layer(
            outputs, filters=filters, layer_num=ii + 1, mode=mode,
            dropout_rate=dropout_rate, dilation_rate=dilation_rate,
        )

    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            outputs, filters=num_classes, kernel_size=1, padding='SAME',
        )
        if save_activations:
            util._add_activation_summary(logits)

    return logits


def _meshnet_model_fn(features, labels, mode, num_classes, dropout_rate=0.25,
                      optimizer='Adam', learning_rate=0.001, config=None):
    """"""
    logits = _meshnet_logit_fn(
        features=features, mode=mode, num_classes=num_classes,
        dropout_rate=dropout_rate,
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


class MeshNet(tf.estimator.Estimator):
    """"""
    def __init__(self, num_classes, model_dir=None, dropout_rate=0.25,
                 optimizer='Adam', learning_rate=0.001, warm_start_from=None,
                 config=None):

        def _model_fn(features, labels, mode, config):
            """"""
            return _meshnet_model_fn(
                features=features, labels=labels, mode=mode,
                num_classes=num_classes, dropout_rate=dropout_rate,
                optimizer=optimizer, learning_rate=learning_rate,
                config=config,
            )

        super(MeshNet, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from,
        )
