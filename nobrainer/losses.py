"""Implementations of loss functions for 3D semantic segmentation."""

import tensorflow as tf
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from nobrainer import metrics


def dice(y_true, y_pred, axis=(1, 2, 3, 4)):
    return 1. - metrics.dice(y_true=y_true, y_pred=y_pred, axis=axis)


class Dice(Loss):
    """Computes one minus the Dice similarity between labels and predictions.
    For example, if `y_true` is [0., 0., 1., 1.] and `y_pred` is [1., 1., 1., 0.]
    then the Dice loss is 0.6. The Dice similarity between these tensors is 0.4.

    Use this loss only for binary semantic segmentation tasks. The default value
    for the axis parameter is meant for models which output a shape of
    `(batch, x, y, z, 1)`. Values in `y_true` and `y_pred` should be in the
    range [0, 1].

    Usage:

    ```python
    dice = nobrainer.losses.Dice(axis=None)
    loss = dice([0., 0., 1., 1.], [1., 1., 1., 0.])
    print('Loss: ', loss.numpy())  # Loss: 0.6
    ```

    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=nobrainer.losses.Dice())
    ```
    """

    def __init__(self,
                 axis=(1, 2, 3, 4),
                 reduction=ReductionV2.SUM_OVER_BATCH_SIZE,
                 name='dice'):
        super(Dice, self).__init__(reduction=reduction, name=name)
        self.axis = axis

    def call(self, y_true, y_pred):
        """Calculates the Dice loss."""
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return dice(y_true=y_true, y_pred=y_pred, axis=self.axis)


def focal(y_true, y_pred):
    raise NotImplementedError()


def focal_tversky(y_true, y_pred):
    """
    https://arxiv.org/pdf/1810.07842.pdf
    """
    raise NotImplementedError()


def generalized_dice(y_true, y_pred, axis=(1, 2, 3)):
    return 1. - metrics.generalized_dice(y_true=y_true, y_pred=y_pred, axis=axis)


class GeneralizedDice(Loss):
    """Computes one minus the generalized Dice similarity between labels and
    predictions. For example, if `y_true` is [[0., 0., 1., 1.]] and `y_pred` is
    [[1., 1., 1., 0.]] then the generalized Dice loss is 0.60. The generalized
    Dice similarity between these tensors is 0.40.

    Use this loss for binary or multi-class semantic segmentation tasks. The
    default value for the axis parameter is meant for models which output a
    shape of `(batch, x, y, z, n_classes)`. Values in `y_true` and `y_pred`
    should be in the range [0, 1].

    Usage:

    ```python
    generalized_dice = nobrainer.losses.GeneralizedDice(axis=1)
    generalized_dice([[0., 0., 1., 1.]], [[1., 1., 1., 0.]])
    print('Loss: ', loss.numpy())  # Loss: 0.60
    ```

    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=nobrainer.losses.GeneralizedDice())
    ```
    """

    def __init__(self,
                 axis=(1, 2, 3),
                 reduction=ReductionV2.SUM_OVER_BATCH_SIZE,
                 name='generalized_dice'):
        super(GeneralizedDice, self).__init__(reduction=reduction, name=name)
        self.axis = axis

    def call(self, y_true, y_pred):
        """Calculates the generalized Dice loss."""
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return generalized_dice(y_true=y_true, y_pred=y_pred, axis=self.axis)


def jaccard(y_true, y_pred, axis=(1, 2, 3, 4)):
    return 1. - metrics.jaccard(y_true=y_true, y_pred=y_pred, axis=axis)


class Jaccard(Loss):
    """Computes one minus the Jaccard similarity between labels and predictions.
    For example, if `y_true` is [0., 0., 1., 1.] and `y_pred` is [1., 1., 1., 0.]
    then the Jaccard loss is 0.75. The Jaccard similarity between these tensors
    is 0.25.

    Use this loss only for binary semantic segmentation tasks. The default value
    for the axis parameter is meant for models which output a shape of
    `(batch, x, y, z, 1)`. Values in `y_true` and `y_pred` should be in the
    range [0, 1].

    Usage:

    ```python
    jaccard = nobrainer.losses.Jaccard(axis=None)
    loss = jaccard([0., 0., 1., 1.], [1., 1., 1., 0.])
    print('Loss: ', loss.numpy())  # Loss: 0.75
    ```

    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=nobrainer.losses.Jaccard())
    ```
    """

    def __init__(self,
                 axis=(1, 2, 3, 4),
                 reduction=ReductionV2.SUM_OVER_BATCH_SIZE,
                 name='jaccard'):
        super(Jaccard, self).__init__(reduction=reduction, name=name)
        self.axis = axis

    def call(self, y_true, y_pred):
        """Calculates the Jaccard loss."""
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return jaccard(y_true=y_true, y_pred=y_pred, axis=self.axis)


def tversky(y_true, y_pred, axis=(1, 2, 3), alpha=0.3, beta=0.7):
    n_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    # Scores are in dynamic range of [0, n_classes].
    scores = metrics.tversky(y_true=y_true, y_pred=y_pred, axis=axis, alpha=alpha, beta=beta)
    return 1. - (scores / n_classes)


class Tversky(Loss):
    """Computes Tversky loss between labels and predictions.

    Use this loss for binary or multi-class semantic segmentation tasks. The
    default value for the axis parameter is meant for binary or multi-class
    predictions. Values in `y_true` and `y_pred` should be in the range [0, 1].
    The default values for alpha and beta are set according to recommendations
    in the Tverky loss manuscript.

    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=nobrainer.losses.Tversky())
    ```
    """

    def __init__(self,
                 axis=(1, 2, 3),
                 alpha=0.3,
                 beta=0.7,
                 reduction=ReductionV2.SUM_OVER_BATCH_SIZE,
                 name='tversky'):
        super(Tversky, self).__init__(reduction=reduction, name=name)
        self.axis = axis
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        """Calculates the Jaccard loss."""
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tversky(y_true=y_true, y_pred=y_pred, axis=self.axis, alpha=self.alpha, beta=self.beta)


def get(loss):
    """Wrapper for `tf.keras.losses.get` that includes Nobrainer's losses."""
    objects = {
        'dice': dice,
        'Dice': Dice,
        'focal': focal,
        'focal_tversky': focal_tversky,
        'jaccard': jaccard,
        'Jaccard': Jaccard,
        'tversky': tversky,
        'Tversky': Tversky,
    }
    with tf.keras.utils.CustomObjectScope(objects):
        return tf.keras.losses.get(loss)