"""QuickNAT implemented in TensorFlow.

Reference
---------
Roy, A. G., Conjeti, S., Navab, N., & Wachinger, C. (2018). QuickNAT:
Segmenting MRI Neuroanatomy in 20 seconds. arXiv preprint arXiv:1801.04161.
"""

import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

FUSED_BATCH_NORM = True

MAX_POOL_KSIZE = (1, 2, 2, 1)
MAX_POOL_STRIDES = (1, 2, 2, 1)


def _dense_block(inputs, block_num, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('batchnorm_{}_1'.format(block_num)):
        bn1 = tf.layers.batch_normalization(
            inputs, training=training, fused=FUSED_BATCH_NORM,
        )
    with tf.variable_scope('relu_{}_1'.format(block_num)):
        relu1 = tf.nn.relu(bn1)
    with tf.variable_scope('conv_{}_1'.format(block_num)):
        conv1 = tf.layers.conv2d(
            relu1, 64, kernel_size=(5, 5), padding='SAME',
        )

    with tf.variable_scope('concat_{}_1'.format(block_num)):
        concat1 = tf.concat([inputs, conv1], axis=-1)

    with tf.variable_scope('batchnorm_{}_2'.format(block_num)):
        bn2 = tf.layers.batch_normalization(
            concat1, training=training, fused=FUSED_BATCH_NORM,
        )
    with tf.variable_scope('relu_{}_2'.format(block_num)):
        relu2 = tf.nn.relu(bn2)
    with tf.variable_scope('conv_{}_2'.format(block_num)):
        conv2 = tf.layers.conv2d(
            relu2, 64, kernel_size=(5, 5), padding='SAME',
        )

    with tf.variable_scope('concat_{}_2'.format(block_num)):
        concat2 = tf.concat([inputs, conv1, conv2], axis=-1)

    with tf.variable_scope('batchnorm_{}_3'.format(block_num)):
        bn3 = tf.layers.batch_normalization(
            concat2, training=training, fused=FUSED_BATCH_NORM,
        )
    with tf.variable_scope('relu_{}_3'.format(block_num)):
        relu3 = tf.nn.relu(bn3)
    with tf.variable_scope('conv_{}_3'.format(block_num)):
        return tf.layers.conv2d(
            relu3, 64, kernel_size=(1, 1), padding='SAME',
        )


# From https://github.com/tensorflow/tensorflow/pull/16885
def unpool_2d(pool,
              ind,
              stride=[1, 2, 2, 1],
              scope='unpool_2d'):
    """Adds a 2D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [
            input_shape[0], input_shape[1] * stride[1],
            input_shape[2] * stride[2], input_shape[3],
        ]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [
            output_shape[0],
            output_shape[1] * output_shape[2] * output_shape[3]
        ]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(
            tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
            shape=[input_shape[0], 1, 1, 1]
        )
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(
            ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64)
        )
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [
            set_input_shape[0], set_input_shape[1] * stride[1],
            set_input_shape[2] * stride[2], set_input_shape[3],
        ]
        ret.set_shape(set_output_shape)
        return ret


def _quicknat_logit_fn(features, num_classes, mode):
    """QuickNAT logit function.

    Parameters
    ----------
    features : numeric Tensor
        Input Tensor.
    num_classes : int
        Number of classes to segment. This is the number of filters in the
        final convolution layer.
    mode : string
        A TensorFlow mode key.

    Returns
    -------
    Tensor of logits.
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    # ENCODING
    with tf.variable_scope('dense_block_1'):
        dense1 = _dense_block(features, block_num=1, mode=mode)
    with tf.variable_scope('maxpool_1'):
        pool1, poolargmax1 = tf.nn.max_pool_with_argmax(
            dense1, ksize=MAX_POOL_KSIZE, strides=MAX_POOL_STRIDES,
            padding='SAME',
        )

    with tf.variable_scope('dense_block_2'):
        dense2 = _dense_block(pool1, block_num=2, mode=mode)
    with tf.variable_scope('maxpool_2'):
        pool2, poolargmax2 = tf.nn.max_pool_with_argmax(
            dense2, ksize=MAX_POOL_KSIZE, strides=MAX_POOL_STRIDES,
            padding='SAME',
        )

    with tf.variable_scope('dense_block_3'):
        dense3 = _dense_block(pool2, block_num=3, mode=mode)
    with tf.variable_scope('maxpool_3'):
        pool3, poolargmax3 = tf.nn.max_pool_with_argmax(
            dense3, ksize=MAX_POOL_KSIZE, strides=MAX_POOL_STRIDES,
            padding='SAME',
        )

    with tf.variable_scope('dense_block_4'):
        dense4 = _dense_block(pool3, block_num=4, mode=mode)
    with tf.variable_scope('maxpool_4'):
        pool4, poolargmax4 = tf.nn.max_pool_with_argmax(
            dense4, ksize=MAX_POOL_KSIZE, strides=MAX_POOL_STRIDES,
            padding='SAME',
        )

    # BOTTLENECK
    with tf.variable_scope('bottleneck'):
        conv_bottleneck = tf.layers.conv2d(
            pool4, 64, kernel_size=(5, 5), padding='SAME'
        )
        bn_bottleneck = tf.layers.batch_normalization(
            conv_bottleneck, training=training
        )

    # DECODING
    with tf.variable_scope('unpool_1'):
        unpool1 = unpool_2d(
            bn_bottleneck, ind=poolargmax4, stride=MAX_POOL_STRIDES
        )

    concat1 = tf.concat([dense4, unpool1], axis=-1)

    with tf.variable_scope('dense_block_5'):
        dense5 = _dense_block(concat1, block_num=5, mode=mode)

    with tf.variable_scope('unpool_2'):
        unpool2 = unpool_2d(dense5, ind=poolargmax3, stride=MAX_POOL_STRIDES)

    concat2 = tf.concat([dense3, unpool2], axis=-1)

    with tf.variable_scope('dense_block_6'):
        dense6 = _dense_block(concat2, block_num=6, mode=mode)

    with tf.variable_scope('unpool_3'):
        unpool3 = unpool_2d(dense6, ind=poolargmax2, stride=MAX_POOL_STRIDES)

    concat3 = tf.concat([dense2, unpool3], axis=-1)

    with tf.variable_scope('dense_block_7'):
        dense7 = _dense_block(concat3, block_num=7, mode=mode)

    with tf.variable_scope('unpool_4'):
        unpool4 = unpool_2d(dense7, ind=poolargmax1, stride=MAX_POOL_STRIDES)

    concat4 = tf.concat([dense1, unpool4], axis=-1)

    with tf.variable_scope('dense_block_8'):
        dense8 = _dense_block(concat4, block_num=8, mode=mode)

    with tf.variable_scope('logits'):
        logits = tf.layers.conv2d(
            dense8, filters=num_classes, kernel_size=(1, 1)
        )

    return logits


def _quicknat_model_fn(features, labels, mode, num_classes, optimizer='Adam',
                       learning_rate=0.001, config=None):
    logits = _quicknat_logit_fn(
        features=features, num_classes=num_classes, mode=mode,
    )
    predictions = tf.argmax(logits, axis=-1)

    optimizer_ = optimizers.get_optimizer_instance(
        optimizer, learning_rate=learning_rate,
    )

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits,
    )
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


class QuickNAT(tf.estimator.Estimator):
    """"""
    def __init__(self, num_classes, model_dir=None, optimizer='Adam',
                 learning_rate=0.001, warm_start_from=None, config=None):

        def _model_fn(features, labels, mode, config):
            """"""
            return _quicknat_model_fn(
                features=features, labels=labels, mode=mode,
                num_classes=num_classes,
                optimizer=optimizer, learning_rate=learning_rate,
                config=config,
            )

        super(QuickNAT, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from,
        )
