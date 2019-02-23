# -*- coding: utf-8 -*-
"""Loss functions implemented in TensorFlow.

Non-differentiable operations (e.g., argmax) are not allowed.

The implementations here are heavily inspired by the official implementations
in `tensorflow.python.ops.losses.losses_impl`.
"""

import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import _safe_div
from tensorflow.python.ops.losses.losses_impl import compute_weighted_loss
from tensorflow.python.ops.losses.losses_impl import Reduction

from nobrainer.util import _EPSILON


def dice(labels, predictions, axis, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Dice loss for binary segmentation. The Dice loss is one minus the Dice
    coefficient, and therefore this loss converges towards zero.

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
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "dice",
                       (predictions, labels, weights)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        intersection = tf.reduce_sum(tf.abs(predictions * labels), axis=axis)
        union = (tf.reduce_sum(predictions, axis=axis) +
                 tf.reduce_sum(labels, axis=axis))
        losses = 1. - ((2 * intersection + _EPSILON) / (union + _EPSILON))
        return compute_weighted_loss(
            losses=losses,
            weights=weights,
            scope=scope,
            loss_collection=loss_collection,
            reduction=reduction)


def generalized_dice(labels, predictions, axis, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Generalized Dice loss for multiclass segmentation.

    Tensors should be one-hot encoded.

    The Generalized Dice loss between labels `g` and predictions `p` is

    .. math::
        1 - 2 \frac{\Sigma_l^C w_l \Sigma_n g_{ln} p_{ln}}
            {\Sigma_l^C w_l \Sigma_n g_{ln} + p_{ln}}

        w_l = \frac{1}{(\Sigma_n^N g_{ln})^2}

    where `C` is the number of classes, and `w_l` is the weight of class `l`.

    References
    ----------
    https://arxiv.org/pdf/1707.03237.pdf
    """
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "generalized_dice",
                       (predictions, labels, weights)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        # Calculate weights per class. Shape of this weight tensor should be
        # `(batch, n_classes)`
        w = tf.reciprocal(tf.square(tf.reduce_sum(labels, axis=axis)) + 1)

        # Outer reduce_sum axis sums across classes.
        num = 2 * tf.reduce_sum(
            w * tf.reduce_sum(tf.abs(labels * predictions), axis=axis), axis=-1)
        # Outer reduce_sum axis sums across classes.
        den = tf.reduce_sum(
            w * tf.reduce_sum(labels + predictions, axis=axis), axis=-1)

        # Shape of losses at this point is (batch,).
        losses = 1 - (num + _EPSILON) / (den + _EPSILON)

        return compute_weighted_loss(
            losses=losses,
            weights=weights,
            scope=scope,
            loss_collection=loss_collection,
            reduction=reduction)


def hamming(labels, predictions, axis, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Hamming loss for binary or multiclass segmentation.

    The Hamming loss is the fraction of unequal samples. This function implements
    a continuous approximation. The Hamming loss between predictions `p` and labels `g`
    is

    .. math::
        g (1 - p) + (1 - g) p

    Parameters
    ----------

    Returns
    -------
    """
    # https://stackoverflow.com/a/45249829/5666087
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "hamming",
                       (predictions, labels, weights)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = tf.reduce_mean(
            labels * (1 - predictions) + (1 - labels) * predictions, axis=axis)
        return compute_weighted_loss(
            losses=losses,
            weights=weights,
            scope=scope,
            loss_collection=loss_collection,
            reduction=reduction)


def jaccard(labels, predictions, axis, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Jaccard distance loss for binary segmentation.

    The Jaccard loss between labels `g` and predictions `p` is

    .. math::
        1 - \frac{|p \cap g| + \epsilon}{|p| + |g| - |p \cap g| + \epsilon}

    """
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "jaccard",
                       (predictions, labels, weights)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        intersection = tf.reduce_sum(tf.abs(predictions * labels), axis=axis)
        denominator = (
            tf.reduce_sum(labels, axis=axis)
            + tf.reduce_sum(predictions, axis=axis)
            - intersection)
        losses = 1 - (intersection + _EPSILON) / (denominator + _EPSILON)
        return compute_weighted_loss(
            losses=losses,
            weights=weights,
            scope=scope,
            loss_collection=loss_collection,
            reduction=reduction)


def tversky(labels, predictions, axis, alpha=0.3, beta=0.7, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Tversky loss for binary or multiclass segmentation.

    Segmentation classes must be in last dimension. Labels should be one-hot
    encoded.

    Identical to Dice loss when alpha = beta = 0.5. The Tversky loss converges
    to `-C`, where `C` is the number of predicted classes.

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
    labels: float `Tensor`, one-hot-encoded ground truth.
    predictions: float `Tensor`
    axis
    alpha: `float`
    beta: `float`, according to the reference, a larger `beta` should emphasize
        recall.

    References
    ----------
    https://arxiv.org/abs/1706.05721
    """
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "tversky",
                       (predictions, labels, weights)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        num = tf.reduce_sum(tf.abs(predictions * labels), axis=axis)
        den = (
            num
            + alpha * tf.reduce_sum(predictions * (1 - labels), axis=axis)
            + beta * tf.reduce_sum((1 - predictions) * labels, axis=axis))

        losses = - tf.reduce_sum((num + _EPSILON) / (den + _EPSILON), axis=-1)
        return compute_weighted_loss(
            losses=losses,
            weights=weights,
            scope=scope,
            loss_collection=loss_collection,
            reduction=reduction)
