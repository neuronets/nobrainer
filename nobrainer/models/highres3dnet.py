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


def _activation_summary(x):
    name = x.op.name + '/activations'
    tf.summary.histogram(name, x)


def _resblock(inputs, filters, kernel_size=3, dilation_rate=1):
    """Return building block of residual network. Residual connections must be
    made outside of this function.

    `inputs -> batchnorm -> relu -> conv -> batchnorm -> relu -> conv`
    """
    with tf.variable_scope('conv1'):
        x1 = tf.layers.batch_normalization(inputs, fused=FUSED_BATCHED_NORM)
        x1 = tf.nn.relu(x1)
        x1 = tf.layers.conv3d(
            x1, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate,
        )
        _activation_summary(x1)
    with tf.variable_scope('conv2'):
        x2 = tf.layers.batch_normalization(x1, fused=FUSED_BATCHED_NORM)
        x2 = tf.nn.relu(x2)
        return tf.layers.conv3d(
            x2, filters=filters, kernel_size=kernel_size, padding='SAME',
            dilation_rate=dilation_rate
        )
        _activation_summary(x2)


def highres3dnet(x, num_classes):
    """Returns logits."""

    with tf.variable_scope('conv1'):
        conv = tf.layers.conv3d(x, filters=16, kernel_size=3, padding='SAME')
        conv = tf.layers.batch_normalization(conv, fused=FUSED_BATCHED_NORM)
        shortcut = tf.nn.relu(conv)
        _activation_summary(shortcut)

    for ii in range(3):
        offset = 1
        name = 'resblock{}'.format(ii + offset)
        with tf.variable_scope(name):
            shortcut = tf.add(
                shortcut, _resblock(shortcut, filters=16)
            )

    for ii in range(3):
        offset = 4
        name = 'resblock{}'.format(ii + offset)
        with tf.variable_scope(name):
            shortcut = tf.add(
                shortcut, _resblock(shortcut, filters=16, dilation_rate=2)
            )

    for ii in range(3):
        offset = 7
        name = 'resblock{}'.format(ii + offset)
        with tf.variable_scope(name):
            shortcut = tf.add(
                shortcut, _resblock(shortcut, filters=16, dilation_rate=4)
            )
    with tf.variable_scope('logits'):
        logits = tf.layers.conv3d(
            shortcut, filters=num_classes, kernel_size=1, padding='SAME'
        )
        _activation_summary(logits)
        return logits
