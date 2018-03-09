"""MeshNet implemented in TensorFlow.

Reference
---------
Fedorov, A., Johnson, J., Damaraju, E., Ozerin, A., Calhoun, V., & Plis, S.
(2017, May). End-to-end learning of brain tissue segmentation from imperfect
labeling. IJCNN 2017. (pp. 3785-3792). IEEE.
"""

import tensorflow as tf
from tensorflow.contrib.estimator import TowerOptimizer, replicate_model_fn
from tensorflow.python.estimator.canned.optimizers import (
    get_optimizer_instance
)

from nobrainer.models.util import check_required_params, set_default_params

FUSED_BATCH_NORM = True


def _layer(inputs,
           mode,
           layer_num,
           filters,
           kernel_size,
           dilation_rate,
           dropout_rate):
    """Layer building block of MeshNet.

    Performs 3D convolution, activation, batch normalization, and dropout on
    `inputs` tensor.

    Args:
        inputs : float `Tensor`, input tensor.
        mode : string, a TensorFlow mode key.
        layer_num : int, value to append to each operator name. This should be
            the layer number in the network.
        filters : int, number of 3D convolution filters.
        kernel_size : int or tuple, size of 3D convolution kernel.
        dilation_rate : int or tuple, rate of dilution in 3D convolution.
        dropout_rate : float, the dropout rate between 0 and 1.

    Returns:
        `Tensor` of same type as `inputs`.
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('layer_{}'.format(layer_num)):
        conv = tf.layers.conv3d(
            inputs, filters=filters, kernel_size=kernel_size,
            padding='SAME', dilation_rate=dilation_rate, activation=None
        )
        activation = tf.nn.relu(conv)
        bn = tf.layers.batch_normalization(
            activation, training=training, fused=FUSED_BATCH_NORM,
        )
        return tf.layers.dropout(bn, rate=dropout_rate, training=training)


def model_fn(features,
             labels,
             mode,
             params):
    """MeshNet model function.

    Args:
        features: 5D float `Tensor`, input tensor. This is the first item
            returned from the `input_fn` passed to `train`, `evaluate`, and
            `predict`. Use `NDHWC` format.
        labels: 4D float `Tensor`, labels tensor. This is the second item
            returned from the `input_fn` passed to `train`, `evaluate`, and
            `predict`. Labels should not be one-hot encoded.
        mode: Optional. Specifies if this training, evaluation or prediction.
        params: `dict` of parameters. All parameters below are required.
            - n_classes: number of classes to classify.
            - optimizer: instance of TensorFlow optimizer.
            - n_filters: number of filters to use in each convolution. The
                original implementation used 21 filters to classify brainmask
                and 71 filters for the multi-class problem.
            - dropout_rate: rate of dropout. For example, 0.1 would drop 10% of
                input units.

    Returns:
        `tf.estimator.EstimatorSpec`

    Raises:
        `ValueError` if required parameters are not in `params`.
    """
    required_params = {'n_classes', 'optimizer'}
    default_params = {'n_filters': 21, 'dropout_rate': 0.25}
    check_required_params(params=params, required_keys=required_params)
    set_default_params(params=params, defaults=default_params)

    tf.logging.debug("Parameters for model:")
    tf.logging.debug(params)

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

    outputs = features

    for ii, dilation_rate in enumerate(dilation_rates):
        outputs = _layer(
            outputs, mode=mode, layer_num=ii + 1, filters=params['n_filters'],
            kernel_size=3, dilation_rate=dilation_rate,
            dropout_rate=params['dropout_rate'],
        )

    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            inputs=outputs, filters=params['n_classes'], kernel_size=(1, 1, 1),
            padding='SAME', activation=None,
        )

    predicted_classes = tf.argmax(logits, axis=-1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # QUESTION (kaczmarj): is this the same as
    # `tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(...))`
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    )

    # Compute metrics here...
    # Use `tf.summary.scalar` to add summaries to tensorboard.

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=None,
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    train_op = params['optimizer'].minimize(
        loss, global_step=tf.train.get_global_step()
    )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


class MeshNet(tf.estimator.Estimator):
    """MeshNet model.

    Args:
        n_classes: int, number of classes to classify.
        optimizer: instance of TensorFlow optimizer or string of optimizer
            name.
        n_filters: int (default 21), number of filters to use in each
            convolution. The original implementation used 21 filters to
            classify brainmask and 71 filters for the multi-class problem.
        dropout_rate: float in range [0, 1] (default 0.25). Rate of dropout.For
            example, 0.1 would drop 10% of input units.
        learning_rate: float, only required if `optimizer` is a string.
        model_dir: Directory to save model parameters, graph, etc. This can
            also be used to load checkpoints from the directory in an estimator
            to continue training a previously saved model. If PathLike object,
            the path will be resolved. If None, the model_dir in config will be
            used if set. If both are set, they must be same. If both are None,
            a temporary directory will be used.
        config: Configuration object.
        warm_start_from: Optional string filepath to a checkpoint to warm-start
            from, or a `tf.estimator.WarmStartSettings` object to fully
            configure warm-starting. If the string filepath is provided instead
            of a `WarmStartSettings`, then all variables are warm-started, and
            it is assumed that vocabularies and Tensor names are unchanged.
        multi_gpu: boolean, if true, optimizer is wrapped in
            `tf.contrib.estimator.TowerOptimizer` and model function is wrapped
            in `tf.contrib.estimator.replicate_model_fn()`.
    """
    def __init__(self,
                 n_classes,
                 optimizer,
                 n_filters=21,
                 dropout_rate=0.25,
                 learning_rate=None,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,
                 multi_gpu=False):
        params = {
            'n_classes': n_classes,
            # If an instance of an optimizer is passed in, this will just
            # return it.
            'optimizer': get_optimizer_instance(optimizer, learning_rate),
            'n_filters': n_filters,
            'dropout_rate': dropout_rate,
        }

        _model_fn = model_fn

        if multi_gpu:
            params['optimizer'] = TowerOptimizer(params['optimizer'])
            _model_fn = replicate_model_fn(_model_fn)

        super(MeshNet, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, params=params,
            config=config, warm_start_from=warm_start_from,
        )
