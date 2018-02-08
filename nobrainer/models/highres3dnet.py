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

FUSED_BATCHED_NORM = True

# TODO: use `tensorflow.python.framework.test_util.is_gpu_available` to
# determine data format. `channels_first` is typically faster on GPU, and
# `channels_last` is typically faster on CPU.


# QUESTION: does adding histograms increase the memory footprint of the model?
def _activation_summary(x):
    name = x.op.name + '/activations'
    tf.summary.histogram(name, x)


def _resblock(inputs, filters, num_prefix, mode, kernel_size=3,
              dilation_rate=1, save_to_histogram=False):
    """Return building block of residual network. Residual connections must be
    made outside of this function.

    `inputs -> batchnorm -> relu -> conv -> batchnorm -> relu -> conv`
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('batchnorm_{}_0'.format(num_prefix)):
        bn1 = tf.layers.batch_normalization(
            inputs, training=training, fused=FUSED_BATCHED_NORM,
        )
    with tf.variable_scope('relu_{}_0'.format(num_prefix)):
        relu1 = tf.nn.relu(bn1)
    with tf.variable_scope('conv_{}_0'.format(num_prefix)):
        conv1 = tf.layers.conv3d(
            relu1, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )
        if save_to_histogram:
            _activation_summary(conv1)

    with tf.variable_scope('batchnorm_{}_1'.format(num_prefix)):
        bn2 = tf.layers.batch_normalization(
            conv1, training=training, fused=FUSED_BATCHED_NORM,
        )
    with tf.variable_scope('relu_{}_1'.format(num_prefix)):
        relu2 = tf.nn.relu(bn2)
    with tf.variable_scope('conv_{}_1'.format(num_prefix)):
        conv2 = tf.layers.conv3d(
            relu2, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )
        if save_to_histogram:
            _activation_summary(conv2)

    with tf.variable_scope('add_{}'.format(num_prefix)):
        return tf.add(conv2, inputs)


def highres3dnet(features, num_classes, mode, save_to_histogram=False):
    """Instantiate HighRes3DNet architecture, and return tensor of logits."""

    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope('conv_0'):
        conv = tf.layers.conv3d(
            features, filters=16, kernel_size=3, padding='SAME'
        )
    with tf.variable_scope('batchnorm_0'):
        conv = tf.layers.batch_normalization(
            conv, training=training, fused=FUSED_BATCHED_NORM
        )
    with tf.variable_scope('relu_0'):
        outputs = tf.nn.relu(conv)

    if save_to_histogram:
        _activation_summary(outputs)

    for ii in range(3):
        offset = 1
        num_prefix = ii + offset
        outputs = _resblock(
            outputs, num_prefix=num_prefix, mode=mode, filters=16
        )

    for ii in range(3):
        offset = 4
        num_prefix = ii + offset
        outputs = _resblock(
            outputs, num_prefix=num_prefix, mode=mode, filters=16,
            dilation_rate=2
        )

    for ii in range(3):
        offset = 7
        num_prefix = ii + offset
        outputs = _resblock(
            outputs, num_prefix=num_prefix, mode=mode, filters=16,
            dilation_rate=4
        )

    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            outputs, filters=num_classes, kernel_size=1, padding='SAME'
        )
        if save_to_histogram:
            _activation_summary(logits)

    return logits
