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
from tensorflow.contrib.estimator import TowerOptimizer, replicate_model_fn
from tensorflow.python.estimator.canned.optimizers import (
    get_optimizer_instance
)

from nobrainer.models.util import check_required_params

FUSED_BATCH_NORM = True

# TODO: use `tensorflow.python.framework.test_util.is_gpu_available` to
# determine data format. `channels_first` is typically faster on GPU, and
# `channels_last` is typically faster on CPU.
# TODO: Once the above todo is implemented, add `channel_format` arguments.


def _resblock(inputs,
              mode,
              layer_num,
              filters,
              kernel_size,
              dilation_rate):
    """Layer building block of residual network. This includes the residual
    connection.

    See the notes below for an overview of the operations performed in this
    function.

    Args:
        inputs : float `Tensor`, input tensor.
        mode : string, TensorFlow mode key.
        layer_num : int, value to append to each operator name. This should be
            the layer number in the network.
        filters : int, number of 3D convolution filters.
        kernel_size : int or tuple, size of 3D convolution kernel.
        dilation_rate : int or tuple, rate of dilution in 3D convolution.

    Returns:
        `Tensor` of same type as `inputs`.

    Notes:
        ```
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
        ```
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('batchnorm_{}_0'.format(layer_num)):
        bn1 = tf.layers.batch_normalization(
            inputs, training=training, fused=FUSED_BATCH_NORM,
        )
    with tf.variable_scope('relu_{}_0'.format(layer_num)):
        relu1 = tf.nn.relu(bn1)
    with tf.variable_scope('conv_{}_0'.format(layer_num)):
        conv1 = tf.layers.conv3d(
            relu1, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )

    with tf.variable_scope('batchnorm_{}_1'.format(layer_num)):
        bn2 = tf.layers.batch_normalization(
            conv1, training=training, fused=FUSED_BATCH_NORM,
        )
    with tf.variable_scope('relu_{}_1'.format(layer_num)):
        relu2 = tf.nn.relu(bn2)
    with tf.variable_scope('conv_{}_1'.format(layer_num)):
        conv2 = tf.layers.conv3d(
            relu2, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )

    with tf.variable_scope('add_{}'.format(layer_num)):
        return tf.add(conv2, inputs)


def model_fn(features,
             labels,
             mode,
             params):
    """HighRes3DNet model function.

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

    Returns:
        `tf.estimator.EstimatorSpec`

    Raises:
        `ValueError` if required parameters are not in `params`.
    """
    required_keys = {'n_classes', 'optimizer'}
    check_required_params(params=params, required_keys=required_keys)

    tf.logging.debug("Parameters for model:")
    tf.logging.debug(params)

    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('conv_0'):
        conv = tf.layers.conv3d(
            features, filters=16, kernel_size=3, padding='SAME')
    with tf.variable_scope('batchnorm_0'):
        conv = tf.layers.batch_normalization(
            conv, training=training, fused=FUSED_BATCH_NORM)
    with tf.variable_scope('relu_0'):
        outputs = tf.nn.relu(conv)

    for ii in range(3):
        offset = 1
        layer_num = ii + offset
        outputs = _resblock(
            outputs, mode=mode, layer_num=layer_num, filters=16, kernel_size=3,
            dilation_rate=1)

    for ii in range(3):
        offset = 4
        layer_num = ii + offset
        outputs = _resblock(
            outputs, mode=mode, layer_num=layer_num, filters=16, kernel_size=3,
            dilation_rate=2)

    for ii in range(3):
        offset = 7
        layer_num = ii + offset
        outputs = _resblock(
            outputs, mode=mode, layer_num=layer_num, filters=16, kernel_size=3,
            dilation_rate=4)

    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            outputs, filters=params['n_classes'], kernel_size=1,
            padding='SAME')

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


class HighRes3DNet(tf.estimator.Estimator):
    """HighRes3DNet model.

    Example:
        ```python
        import numpy as np
        import tensorflow as tf

        shape = (1, 10, 10, 10)  # Batch of 1.
        X = np.random.rand(*shape, 1).astype(np.float32)
        y = np.random.randint(0, 9, size=(shape), dtype=np.int32)
        dset_fn = lambda: tf.data.Dataset.from_tensors((X, y))
        estimator = nobrainer.models.HighRes3DNet(
            n_classes=10, optimizer='Adam', learning_rate=0.001,
        )
        estimator.train(input_fn=dset_fn)
        ```

    Args:
        n_classes: int, number of classes to classify.
        optimizer: instance of TensorFlow optimizer or string of optimizer
            name.
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
        }

        _model_fn = model_fn

        if multi_gpu:
            params['optimizer'] = TowerOptimizer(params['optimizer'])
            _model_fn = replicate_model_fn(_model_fn)

        super(HighRes3DNet, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, params=params,
            config=config, warm_start_from=warm_start_from,
        )
