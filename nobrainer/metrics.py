"""Implementations of metrics for 3D semantic segmentation."""

import tensorflow as tf


def average_volume_difference():
    raise NotImplementedError()


def dice(y_true, y_pred, axis=(1, 2, 3, 4)):
    """Calculate Dice similarity between labels and predictions.

    Dice similarity is in [0, 1], where 1 is perfect overlap and 0 is no
    overlap. If both labels and predictions are empty (e.g., all background),
    then Dice similarity is 1.

    If we assume the inputs are rank 5 [`(batch, x, y, z, classes)`], then an
    axis parameter of `(1, 2, 3)` will result in a tensor that contains a Dice
    score for every class in every item in the batch. The shape of this tensor
    will be `(batch, classes)`. If the inputs only have one class (e.g., binary
    segmentation), then an axis parameter of `(1, 2, 3, 4)` should be used.
    This will result in a tensor of shape `(batch,)`, where every value is the
    Dice similarity for that prediction.

    Implemented according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ6

    Returns
    -------
    Tensor of Dice similarities.

    Citations
    ---------
    Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation:
        analysis, selection, and tool. BMC Med Imaging. 2015;15:29. Published 2015
        Aug 12. doi:10.1186/s12880-015-0068-x
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    eps = tf.keras.backend.epsilon()

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    summation = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
    return (2 * intersection + eps) / (summation + eps)


def generalized_dice(y_true, y_pred, axis=(1, 2, 3)):
    """Calculate Generalized Dice similarity. This is useful for multi-class
    predictions.

    If we assume the inputs are rank 5 [`(batch, x, y, z, classes)`], then an
    axis parameter of `(1, 2, 3)` should be used. This will result in a tensor
    of shape `(batch,)`, where every value is the Generalized Dice similarity
    for that prediction, across all classes.

    Returns
    -------
    Tensor of Generalized Dice similarities.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    if y_true.get_shape().ndims < 2 or y_pred.get_shape().ndims < 2:
        raise ValueError("y_true and y_pred must be at least rank 2.")

    epsilon = tf.keras.backend.epsilon()
    w = tf.math.reciprocal(tf.square(tf.reduce_sum(y_true, axis=axis)) + epsilon)
    num = 2 * tf.reduce_sum(w * tf.reduce_sum(y_true * y_pred, axis=axis), axis=-1)
    den = tf.reduce_sum(w * tf.reduce_sum(y_true + y_pred, axis=axis), axis=-1)
    return (num + epsilon) / (den + epsilon)


def hamming(y_true, y_pred, axis=(1, 2, 3)):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.not_equal(y_pred, y_true), axis=axis)


def haussdorf():
    raise NotADirectoryError()


def jaccard(y_true, y_pred, axis=(1, 2, 3, 4)):
    """Calculate Jaccard similarity between labels and predictions.

    Jaccard similarity is in [0, 1], where 1 is perfect overlap and 0 is no
    overlap. If both labels and predictions are empty (e.g., all background),
    then Jaccard similarity is 1.

    If we assume the inputs are rank 5 [`(batch, x, y, z, classes)`], then an
    axis parameter of `(1, 2, 3)` will result in a tensor that contains a Jaccard
    score for every class in every item in the batch. The shape of this tensor
    will be `(batch, classes)`. If the inputs only have one class (e.g., binary
    segmentation), then an axis parameter of `(1, 2, 3, 4)` should be used.
    This will result in a tensor of shape `(batch,)`, where every value is the
    Jaccard similarity for that prediction.

    Implemented according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ7

    Returns
    -------
    Tensor of Jaccard similarities.

    Citations
    ---------
    Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation:
        analysis, selection, and tool. BMC Med Imaging. 2015;15:29. Published 2015
        Aug 12. doi:10.1186/s12880-015-0068-x
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    eps = tf.keras.backend.epsilon()

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
    return (intersection + eps) / (union - intersection + eps)


def tversky(y_true, y_pred, axis=(1, 2, 3), alpha=0.3, beta=0.7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    if y_true.get_shape().ndims < 2 or y_pred.get_shape().ndims < 2:
        raise ValueError("y_true and y_pred must be at least rank 2.")

    eps = tf.keras.backend.epsilon()

    num = tf.reduce_sum(y_pred * y_true, axis=axis)
    den = (
        num
        + alpha * tf.reduce_sum(y_pred * (1 - y_true), axis=axis)
        + beta * tf.reduce_sum((1 - y_pred) * y_true, axis=axis)
    )
    # Sum over classes.
    return tf.reduce_sum((num + eps) / (den + eps), axis=-1)
