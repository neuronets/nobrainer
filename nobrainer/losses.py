# -*- coding: utf-8 -*-
"""Loss functions implemented in TensorFlow.

Non-differentiable operations (e.g., type casting, argmax) are not allowed.
"""

# TODO(kaczmarj): write losses like https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/losses/losses_impl.py
# TODO(kaczmarj): write tests!
# TODO(kaczmarj): generalized dice, Lovász-Softmax loss
# Use https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
# as a reference for Lovász-Softmax.

import tensorflow as tf

_ONE = tf.constant(1.0, dtype=tf.float32)
_TWO = tf.constant(2.0, dtype=tf.float32)
_EPSILON = tf.constant(1e-7, dtype=tf.float32)


def dice(labels, predictions):
    """Return the Dice loss between labels and predictions. The Dice loss is
    one minus the Dice coefficient, and therefore this loss converges towards
    zero.

    The Dice loss between predictions `p` and labels `g` is

    .. math::

        1 - \frac{2 \Sigma_i^N p_i g_i + \epsilon}
            {\Sigma_i^N p_i^2 + \Sigma_i^N g_i^2 + \epsilon}

    where `\epsilon` is a small value for stability.

    Parameters
    ----------
    labels: float `Tensor`
    predictions: float `Tensor`

    References
    ----------
    https://arxiv.org/pdf/1606.04797.pdf
    """
    num = _TWO * tf.reduce_sum(predictions * labels)
    den = (
        tf.reduce_sum(tf.square(predictions))
        + tf.reduce_sum(tf.square(labels)))
    return _ONE - (num + _EPSILON) / (den + _EPSILON)


def generalized_dice(labels, predictions):
    """Generalized Dice loss.

    .. math::
        w_l = \frac{1}{(\Sigma_n^Nr_{ln})^2}

    References
    ----------
    https://arxiv.org/pdf/1707.03237.pdf
    """
    # Weights per label. Arrays should probably be one-hot encoded.
    # Weights should have shape (batch, n_classes)
    axis = (0, 1, 2, 3)
    weights = tf.reciprocal(tf.square(tf.reduce_sum(labels, axis=axis)))

    num = _TWO * tf.reduce_sum(
        weights * tf.reduce_sum(labels * predictions, axis=axis))

    den = tf.reduce_sum(
        weights * tf.reduce_sum(labels + predictions, axis=axis))
    return 1 - (num + _EPSILON) / (den + _EPSILON)


def jaccard(labels, predictions):
    """Jaccard loss.

    The Jaccard loss between predictions `p` and labels `g` is

    .. math::
        1 - \frac{p \cup g + \epsilon}{p \cap g + \epsilon}

    """
    num = tf.reduce_sum(predictions * labels)
    den = tf.reduce_sum(predictions + labels) - num
    return _ONE - (num + _EPSILON) / (den + _EPSILON)


def tversky(labels, predictions, alpha=0.3, beta=0.7):
    """Return the Tversky loss between labels and predictions.

    The Tversky loss between predictions `p` and labels `g` is

    .. math::
        \frac{\Sigma_i^N p_{0i} g_{0i} + \epsilon}
            {\Sigma_i^N p_{0i} g_{0i}
             + \alpha \Sigma_i^N p_{0i} g_{1i}
             + \beta \Sigma_i^N p_{1i} g_{0i}
             + \epsilon}

    where `p_{0i}` is the probability that voxel `i` belongs to foreground,
    `p_{1i}` is the probability that voxel `i` belongs to background, `g_{0i}`
    is the probability that voxel `i` of the ground truth belongs to foreground,
    `g_{1i}` is the probability that voxel `i` of the ground truth belongs
    to background, and `\epsilon` is a small value for stability.

    Parameters
    ----------
    labels: float `Tensor`, ground
    predictions: float `Tensor`
    alpha: `float`
    beta: `float`, a larger `beta` should emphasize recall.

    References
    ----------
    https://arxiv.org/abs/1706.05721
    """
    num = tf.reduce_sum(predictions * labels)
    den = (
        tf.reduce_sum(predictions * labels)
        + alpha * tf.reduce_sum(predictions * (_ONE - labels))
        + beta * tf.reduce_sum((_ONE - predictions) * labels))
    return (num + _EPSILON) / (den + _EPSILON)
