"""MeshNet implemented in TensorFlow.

Reference
---------
Fedorov, A., Johnson, J., Damaraju, E., Ozerin, A., Calhoun, V., & Plis, S.
(2017, May). End-to-end learning of brain tissue segmentation from imperfect
labeling. IJCNN 2017. (pp. 3785-3792). IEEE.
"""

import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

from nobrainer.metrics import dice_coefficient_by_class_numpy
from nobrainer.models import util

FUSED_BATCH_NORM = True


def _layer(inputs, layer_num, mode, filters, dropout_rate, kernel_size=3,
           dilation_rate=1):
    """Return layer building block of MeshNet.

    See the notes below for an overview of the operations performed in this
    function.

    Args:
        inputs : float `Tensor`, input tensor.
        layer_num : int, value to append to each operator name. This should be
        the layer number in the network.
        mode : string, a TensorFlow mode key.
        filters : int, number of 3D convolution filters.
        dropout_rate : float, the dropout rate between 0 and 1.
        kernel_size : int or tuple, size of 3D convolution kernel.
        dilation_rate : int or tuple, rate of dilution in 3D convolution.

    Returns:
        `Tensor` of sample type as `inputs`.

    Notes:
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
            relu, training=training, fused=FUSED_BATCH_NORM,
        )
        dropout = tf.layers.dropout(bn, rate=dropout_rate, training=training)

        util._add_activation_summary(dropout)

        return dropout


def _meshnet_logit_fn(features, num_classes, mode, dropout_rate=0.25):
    """MeshNet logit function.

    Args:
        features : float `Tensor`, input tensor.
        num_classes : int, number of classes to segment. This is the number of
            filters in the final convolutional layer.
        mode : string, a TensorFlow mode key.
        dropout_rate : float, the dropout rate between 0 and 1.

    Returns:
        `Tensor` of logits.
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

    # Add Dice coefficients to summary for visualization in TensorBoard.
    predictions_onehot = tf.one_hot(predictions, depth=num_classes)
    labels_onehot = tf.one_hot(labels, depth=num_classes)

    dice_coefs = tf.py_func(
        dice_coefficient_by_class_numpy, [labels_onehot, predictions_onehot],
        tf.float32,
    )
    for ii in range(num_classes):
        tf.summary.scalar('dice_label{}'.format(ii), dice_coefs[ii])

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
