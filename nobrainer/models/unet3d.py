# -*- coding: utf-8 -*-
"""Three-dimensional U-Net implemented in TensorFlow.

Reference
---------
Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016)
3D U-Net: learning dense volumetric segmentation from sparse annotation.
International Conference on Medical Image Computing and Computer-Assisted
Intervention (pp. 424-432). Springer, Cham.

PDF available at https://arxiv.org/pdf/1606.06650.pdf.
"""

import tensorflow as tf
from tensorflow.contrib.estimator import TowerOptimizer
from tensorflow.contrib.estimator import replicate_model_fn
from tensorflow.python.estimator.canned.optimizers import (
    get_optimizer_instance
)

from nobrainer import losses
from nobrainer import metrics
from nobrainer.models.util import check_optimizer_for_training
from nobrainer.models.util import check_required_params
from nobrainer.models.util import set_default_params

FUSED_BATCH_NORM = True
_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)


def _conv_block(x, filters1, filters2, mode, layer_num, batchnorm=True):
    """Convolution block.

    Args:
        x: float `Tensor`, input tensor.
        filters1: int, output space of first convolution.
        filters2: int, output space of second convolution.
        mode: string, TensorFlow mode key.
        batchnorm: bool, if true, apply batch normalization after each
            convolution.

    Returns:
        `Tensor` of sample type as `x`.

    Notes:
    ```
    inputs -> conv3d -> [batchnorm] -> relu -> conv3d -> [batchnorm] -> relu
    ```
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('conv_{}_0'.format(layer_num)):
        x = tf.layers.conv3d(
            inputs=x, filters=filters1, kernel_size=(3, 3, 3), padding='SAME', kernel_regularizer=_regularizer)

    if batchnorm:
        with tf.variable_scope('batchnorm_{}_0'.format(layer_num)):
            x = tf.layers.batch_normalization(
                inputs=x, training=training, fused=FUSED_BATCH_NORM)

    with tf.variable_scope('relu_{}_0'.format(layer_num)):
        x = tf.nn.relu(x)

    with tf.variable_scope('conv_{}_1'.format(layer_num)):
        x = tf.layers.conv3d(
            inputs=x, filters=filters2, kernel_size=(3, 3, 3), padding='SAME', kernel_regularizer=_regularizer)

    if batchnorm:
        with tf.variable_scope('batchnorm_{}_1'.format(layer_num)):
            x = tf.layers.batch_normalization(
                inputs=x, training=training, fused=FUSED_BATCH_NORM)

    with tf.variable_scope('relu_{}_1'.format(layer_num)):
        x = tf.nn.relu(x)

    return x


def model_fn(features, labels, mode, params, config=None):
    """3D U-Net model function.

    Args:

    Returns:

    Raises:
    """
    volume = features
    if isinstance(volume, dict):
        volume = features['volume']

    required_keys = {'n_classes'}
    default_params ={
        'optimizer': None,
        'batchnorm': True,
    }

    check_required_params(params=params, required_keys=required_keys)
    set_default_params(params=params, defaults=default_params)
    check_optimizer_for_training(optimizer=params['optimizer'], mode=mode)

    bn = params['batchnorm']

    # start encoding
    shortcut_1 = _conv_block(
        volume, filters1=32, filters2=64, mode=mode, layer_num=0, batchnorm=bn)

    with tf.variable_scope('maxpool_1'):
        x = tf.layers.max_pooling3d(
            inputs=shortcut_1, pool_size=(2, 2, 2), strides=(2, 2, 2),
            padding='same')

    shortcut_2 = _conv_block(
        x, filters1=64, filters2=128, mode=mode, layer_num=1, batchnorm=bn)

    with tf.variable_scope('maxpool_2'):
        x = tf.layers.max_pooling3d(
            inputs=shortcut_2, pool_size=(2, 2, 2), strides=(2, 2, 2),
            padding='same')

    shortcut_3 = _conv_block(
        x, filters1=128, filters2=256, mode=mode, layer_num=2, batchnorm=bn)

    with tf.variable_scope('maxpool_3'):
        x = tf.layers.max_pooling3d(
            inputs=shortcut_3, pool_size=(2, 2, 2), strides=(2, 2, 2),
            padding='same')

    x = _conv_block(
        x, filters1=256, filters2=512, mode=mode, layer_num=3, batchnorm=bn)

    # start decoding
    with tf.variable_scope("upconv_0"):
        x = tf.layers.conv3d_transpose(
            inputs=x, filters=512, kernel_size=(2, 2, 2), strides=(2, 2, 2), kernel_regularizer=_regularizer)

    with tf.variable_scope('concat_1'):
        x = tf.concat((shortcut_3, x), axis=-1)

    x = _conv_block(
        x, filters1=256, filters2=256, mode=mode, layer_num=4, batchnorm=bn)

    with tf.variable_scope("upconv_1"):
        x = tf.layers.conv3d_transpose(
            inputs=x, filters=256, kernel_size=(2, 2, 2), strides=(2, 2, 2), kernel_regularizer=_regularizer)

    with tf.variable_scope('concat_2'):
        x = tf.concat((shortcut_2, x), axis=-1)

    x = _conv_block(
        x, filters1=128, filters2=128, mode=mode, layer_num=5, batchnorm=bn)

    with tf.variable_scope("upconv_2"):
        x = tf.layers.conv3d_transpose(
            inputs=x, filters=128, kernel_size=(2, 2, 2), strides=(2, 2, 2), kernel_regularizer=_regularizer)

    with tf.variable_scope('concat_3'):
        x = tf.concat((shortcut_1, x), axis=-1)

    x = _conv_block(
        x, filters1=64, filters2=64, mode=mode, layer_num=6, batchnorm=bn)

    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            inputs=x, filters=params['n_classes'], kernel_size=(1, 1, 1),
            padding='same', activation=None, kernel_regularizer=_regularizer)
    # end decoding

    with tf.variable_scope('predictions'):
        predictions = tf.nn.softmax(logits=logits)

    class_ids = tf.argmax(logits, axis=-1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': class_ids,
            'probabilities': predictions,
            'logits': logits}
        # Outputs for SavedModel.
        export_outputs = {
            'outputs': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    onehot_labels = tf.one_hot(labels, params['n_classes'])

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=onehot_labels, logits=logits)
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={
                'dice': metrics.streaming_dice(
                    labels, class_ids, axis=(1, 2, 3)),
            })

    assert mode == tf.estimator.ModeKeys.TRAIN

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = params['optimizer'].minimize(loss, global_step=tf.train.get_global_step())

    dice_coefficients = tf.reduce_mean(
        metrics.dice(
            onehot_labels,
            tf.one_hot(class_ids, axis=(1, 2, 3)),
        axis=0))

    logging_hook = tf.train.LoggingTensorHook(
        {"loss" : loss, "dice": dice_coefficients}, every_n_iter=100)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[logging_hook])


class UNet3D(tf.estimator.Estimator):
    """Three-dimensional U-Net model.
    """
    def __init__(self,
                 n_classes,
                 optimizer=None,
                 learning_rate=None,
                 batchnorm=True,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,
                 multi_gpu=False):
        params = {
            'n_classes': n_classes,
            # If an instance of an optimizer is passed in, this will just
            # return it.
            'optimizer': (
                None if optimizer is None
                else get_optimizer_instance(optimizer, learning_rate)),
            'batchnorm': batchnorm,
        }

        _model_fn = model_fn

        if multi_gpu:
            params['optimizer'] = TowerOptimizer(params['optimizer'])
            _model_fn = replicate_model_fn(_model_fn)

        super(UNet3D, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, params=params,
            config=config, warm_start_from=warm_start_from)
