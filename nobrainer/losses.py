"""Implementations of loss functions for 3D semantic segmentation."""

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from . import metrics


def dice(y_true, y_pred, axis=(1, 2, 3, 4)):
    return 1.0 - metrics.dice(y_true=y_true, y_pred=y_pred, axis=axis)


class Dice(LossFunctionWrapper):
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

    def __init__(self, axis=(1, 2, 3, 4), reduction=ReductionV2.AUTO, name="dice"):
        super().__init__(dice, axis=axis, reduction=reduction, name=name)


def focal_tversky(y_true, y_pred):
    """
    https://arxiv.org/pdf/1810.07842.pdf
    """
    raise NotImplementedError()


def generalized_dice(y_true, y_pred, axis=(1, 2, 3)):
    return 1.0 - metrics.generalized_dice(y_true=y_true, y_pred=y_pred, axis=axis)


class GeneralizedDice(LossFunctionWrapper):
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

    def __init__(
        self, axis=(1, 2, 3), reduction=ReductionV2.AUTO, name="generalized_dice"
    ):
        super().__init__(generalized_dice, axis=axis, reduction=reduction, name=name)


def jaccard(y_true, y_pred, axis=(1, 2, 3, 4)):
    return 1.0 - metrics.jaccard(y_true=y_true, y_pred=y_pred, axis=axis)


class Jaccard(LossFunctionWrapper):
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

    def __init__(self, axis=(1, 2, 3, 4), reduction=ReductionV2.AUTO, name="jaccard"):
        super().__init__(jaccard, axis=axis, reduction=reduction, name=name)


def tversky(y_true, y_pred, axis=(1, 2, 3), alpha=0.3, beta=0.7):
    n_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    # Scores are in dynamic range of [0, n_classes].
    scores = metrics.tversky(
        y_true=y_true, y_pred=y_pred, axis=axis, alpha=alpha, beta=beta
    )
    return 1.0 - (scores / n_classes)


class Tversky(LossFunctionWrapper):
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

    def __init__(
        self,
        axis=(1, 2, 3),
        alpha=0.3,
        beta=0.7,
        reduction=ReductionV2.AUTO,
        name="tversky",
    ):
        super().__init__(
            tversky, axis=axis, alpha=alpha, beta=beta, reduction=reduction, name=name
        )


def elbo(y_true, y_pred, model, num_examples, from_logits=False):
    """Labels should be integers in `[0, n)`."""
    scc_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    neg_log_likelihood = scc_fn(y_true, y_pred)
    kl = sum(model.losses) / num_examples
    elbo_loss = neg_log_likelihood + kl
    return elbo_loss


class ELBO(LossFunctionWrapper):
    """Loss to minimize Evidence Lower Bound (ELBO).

    Use this loss for multiclass variational segmentation.
    Labels should not be one-hot encoded.
    """

    def __init__(
        self,
        model,
        num_examples,
        from_logits=False,
        reduction=ReductionV2.AUTO,
        name="ELBO",
    ):
        super().__init__(
            elbo,
            model=model,
            num_examples=num_examples,
            from_logits=from_logits,
            name=name,
            reduction=reduction,
        )


def wasserstein(y_true, y_pred):
    return y_true * y_pred


class Wasserstein(LossFunctionWrapper):
    """Computes Wasserstein loss between labels and predictions.

    Aims to score the realness or fakeness of a given image and measures the Earth Mover (EM)
    distance. Use this loss for training GANs. Values for `y_true` is 1 or -1 for real and fake,
    and `y_pred` is from discriminator. Use in combination with gradient clipping or gradient
    penalty (WassersteinGP defined below).
    https://arxiv.org/abs/1701.07875

    Usage:
    ```python
    real_pred = discriminator(real)
    fake_pred = discriminator(fake)

    wasserstein_loss = Wasserstein()
    disciminator_loss = wasserstein_loss(1, real_pred) + wasserstein_loss(-1, fake_pred)
    generator_loss = wasserstein_loss(1, fake_pred)
    ```
    """

    def __init__(self, reduction=ReductionV2.AUTO, name="wasserstein"):
        super().__init__(wasserstein, reduction=reduction, name=name)


def gradient_penalty(gradients, real_pred, gp_weight=10, epsilon_weight=0.001):

    gradients_squared = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(
        gradients_squared, axis=tf.range(1, tf.rank(gradients_squared))
    )
    gradient_l2_norm = tf.sqrt(gradients_sqr_sum)

    gradient_penalty = gp_weight * tf.square(1 - gradient_l2_norm)

    epsilon_loss = epsilon_weight * tf.square(real_pred)

    return gradient_penalty + epsilon_loss


class GradientPenalty(LossFunctionWrapper):
    """Computes Gradient Penalty for Wasserstein Loss.
    Improvement to gradient clipping for Wasserstein-GAN to enforce the Lipschitz constraint.
    Uses the points interpolated between real and fake to ensure a gradient norm of 1.
    https://arxiv.org/pdf/1704.00028.pdf

    Usage:
    ```python
    fakes = generator(latents)

    wasserstein_gp = WassersteinGP(discriminator=discriminator, alpha=alpha)
    gradient_penalty = wasserstein_gp(reals, fakes)
    ```
    """

    def __init__(
        self,
        gp_weight=10,
        epsilon_weight=0.001,
        reduction=ReductionV2.AUTO,
        name="wasserstein_gp",
    ):
        super().__init__(
            gradient_penalty,
            gp_weight=gp_weight,
            epsilon_weight=epsilon_weight,
            reduction=reduction,
            name=name,
        )


def get(loss):
    """Wrapper for `tf.keras.losses.get` that includes Nobrainer's losses."""
    objects = {
        "dice": dice,
        "Dice": Dice,
        "focal_tversky": focal_tversky,
        "jaccard": jaccard,
        "Jaccard": Jaccard,
        "tversky": tversky,
        "Tversky": Tversky,
        "elbo": elbo,
        "ELBO": ELBO,
        "wasserstein": wasserstein,
        "Wasserstein": Wasserstein,
        "gradient_penalty": gradient_penalty,
        "GradientPenalty": GradientPenalty,
    }
    with tf.keras.utils.CustomObjectScope(objects):
        return tf.keras.losses.get(loss)
