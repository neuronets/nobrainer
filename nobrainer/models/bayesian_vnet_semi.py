# Model definition for a Semi-Bayesian VNet with deterministic
# encoder and Bayesian decoder
from tensorflow.keras.layers import (
    Conv3D,
    Input,
    MaxPooling3D,
    UpSampling3D,
    concatenate,
)
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

from ..bayesian_utils import prior_fn_for_bayesian
from ..layers.groupnorm import GroupNormalization

tfd = tfp.distributions


def down_stage(inputs, filters, kernel_size=3, activation="relu", padding="SAME"):
    """encoding blocks of the Semi-Bayesian VNet model.

    Parameters
    ----------
    inputs: tf.layer for encoding stage.
    filters: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Returns
    ----------
    encoding module
    """
    conv = Conv3D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv = GroupNormalization()(conv)
    conv = Conv3D(filters, kernel_size, activation=activation, padding=padding)(conv)
    conv = GroupNormalization()(conv)
    pool = MaxPooling3D()(conv)
    return conv, pool


def up_stage(
    inputs,
    skip,
    filters,
    prior_fn,
    kernel_posterior_fn,
    kld,
    kernel_size=3,
    activation="relu",
    padding="SAME",
):
    """decoding blocks of the Semi-Bayesian VNet model.

    Parameters
    ----------
    inputs: tf.layer for encoding stage.
    skip: setting skip connections
    kld: a func to compute KL Divergence loss, default is set None.
        KLD can be set as (lambda q, p, ignore: kl_lib.kl_divergence(q, p))
    prior_fn: a func to initialize priors distributions
    kernel_posterior_fn:a func to initlaize kernel posteriors
        (loc, scale and weightnorms)
    filters: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Returns
    ----------
    decoding module.
    """
    up = UpSampling3D()(inputs)
    up = tfp.layers.Convolution3DFlipout(
        filters,
        2,
        activation=activation,
        padding=padding,
        kernel_divergence_fn=kld,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_prior_fn=prior_fn,
    )(up)
    up = GroupNormalization()(up)

    merge = concatenate([skip, up])
    merge = GroupNormalization()(merge)

    conv = tfp.layers.Convolution3DFlipout(
        filters,
        kernel_size,
        activation=activation,
        padding=padding,
        kernel_divergence_fn=kld,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_prior_fn=prior_fn,
    )(merge)
    conv = GroupNormalization()(conv)
    conv = tfp.layers.Convolution3DFlipout(
        filters,
        kernel_size,
        activation=activation,
        padding=padding,
        kernel_divergence_fn=kld,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_prior_fn=prior_fn,
    )(conv)
    conv = GroupNormalization()(conv)

    return conv


def end_stage(
    inputs,
    prior_fn,
    kernel_posterior_fn,
    kld,
    n_classes=1,
    kernel_size=3,
    activation="relu",
    padding="SAME",
):
    """last logit layer of Semi-Bayesian VNet.

    Parameters
    ----------
    inputs: tf.model layer.
    kld: a func to compute KL Divergence loss, default is set None.
        KLD can be set as (lambda q, p, ignore: kl_lib.kl_divergence(q, p))
    prior_fn: a func to initialize priors distributions
    kernel_posterior_fn:a func to initlaize kernel posteriors
        (loc, scale and weightnorms)
    n_classes: int, for binary class use the value 1.
    kernal_size: int, size of the kernel of conv layers. Default kernel size
        is set to be 3.
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use

    Result
    ----------
    prediction probabilities.
    """
    conv = tfp.layers.Convolution3DFlipout(
        n_classes,
        kernel_size,
        activation=activation,
        padding="SAME",
        kernel_divergence_fn=kld,
        kernel_posterior_fn=kernel_posterior_fn,
        kernel_prior_fn=prior_fn,
    )(inputs)
    if n_classes == 1:
        conv = tfp.layers.Convolution3DFlipout(
            n_classes,
            1,
            activation="sigmoid",
            kernel_divergence_fn=kld,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_prior_fn=prior_fn,
        )(conv)
    else:
        conv = tfp.layers.Convolution3DFlipout(
            n_classes,
            1,
            activation="softmax",
            kernel_divergence_fn=kld,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_prior_fn=prior_fn,
        )(conv)
    return conv


def bayesian_vnet_semi(
    n_classes=1,
    input_shape=(256, 256, 256, 1),
    kernel_size=3,
    prior_fn=prior_fn_for_bayesian(),
    kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
    kld=None,
    activation="relu",
    padding="SAME",
):
    """Instantiate a 3D Semi-Bayesian VNet Architecture.

    Adapted from Deterministic VNet: https://arxiv.org/pdf/1606.04797.pdf
    Encoder has 3D Convolutional layers and Decoder has 3D
    Flipout(variational layers).

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    kernal_size(int): size of the kernel of conv layers
    activation(str): all tf.keras.activations are allowed
    kld: a func to compute KL Divergence loss, default is set None.
        KLD can be set as (lambda q, p, ignore: kl_lib.kl_divergence(q, p))
    prior_fn: a func to initialize priors distributions
    kernel_posterior_fn:a func to initlaize kernel posteriors
        (loc, scale and weightnorms)
    See Bayesian Utils for more options for kld, prior_fn and kernal_posterior_fn
    activation: str or optimizer object, the non-linearity to use. All
        tf.activations are allowed to use.

    Returns
    ----------
    Bayesian model object.
    """
    inputs = Input(input_shape)

    conv1, pool1 = down_stage(
        inputs, 16, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv2, pool2 = down_stage(
        pool1, 32, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv3, pool3 = down_stage(
        pool2, 64, kernel_size=kernel_size, activation=activation, padding=padding
    )
    conv4, _ = down_stage(
        pool3, 128, kernel_size=kernel_size, activation=activation, padding=padding
    )

    conv5 = up_stage(
        conv4,
        conv3,
        64,
        prior_fn,
        kernel_posterior_fn,
        kld,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )
    conv6 = up_stage(
        conv5,
        conv2,
        32,
        prior_fn,
        kernel_posterior_fn,
        kld,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )
    conv7 = up_stage(
        conv6,
        conv1,
        16,
        prior_fn,
        kernel_posterior_fn,
        kld,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )

    conv8 = end_stage(
        conv7,
        prior_fn,
        kernel_posterior_fn,
        kld,
        n_classes=n_classes,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
    )

    return Model(inputs=inputs, outputs=conv8)
