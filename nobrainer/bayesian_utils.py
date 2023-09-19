import numpy as np
import tensorflow as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.ops import nn_impl
import tensorflow_probability as tfp
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.distributions import (
    deterministic as deterministic_lib,
)
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib

tfd = tfp.distributions


def default_loc_scale_fn(
    is_singular=True,
    loc_initializer=tf.keras.initializers.he_normal(),
    untransformed_scale_initializer=tf.constant_initializer(0.0001),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None,
    weightnorm=False,
):
    """Function for `mean`, `std`, 'weightnorm' tf.variables.

    This function produces a closure which produces `mean`, `std`, 'weightnorm'
    using`tf.get_variable`. The closure accepts the following arguments:
    dtype: Type of parameter's event.
    shape: Python `list`-like representing the parameter's event shape.
    name: Python `str` name prepended to any created (or existing)
     `tf.Variable`s.
    trainable: Python `bool` indicating all created `tf.Variable`s should be
     added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
    add_variable_fn: `tf.get_variable`-like `callable` used to create (or
     access existing) `tf.Variable`s.

    Parameters
    ----------
        is_singular: Python `bool` indicating if `scale is None`. Default: `False`.
        loc_initializer: Initializer function for the `loc` parameters.
          The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
        untransformed_scale_initializer: Initializer function for the `scale`
          parameters. Default value: `tf.random_normal_initializer(mean=-3.,
          stddev=0.1)`. This implies the softplus transformed result is initialized
          near `0`. It allows a `Normal` distribution with `scale` parameter set to
          this value to approximately act like a point mass.
        loc_regularizer: Regularizer function for the `loc` parameters.
          The default (`None`) is to use the `tf.get_variable` default.
        untransformed_scale_regularizer: Regularizer function for the `scale`
          parameters. The default (`None`) is to use the `tf.get_variable` default.
        loc_constraint: An optional projection function to be applied to the
          loc after being updated by an `Optimizer`. The function must take as input
          the unprojected variable and must return the projected variable (which
          must have the same shape). Constraints are not safe to use when doing
          asynchronous distributed training.
          The default (`None`) is to use the `tf.get_variable` default.
        untransformed_scale_constraint: An optional projection function to be
          applied to the `scale` parameters after being updated by an `Optimizer`
          (e.g. used to implement norm constraints or value constraints). The
          function must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are not
          safe to use when doing asynchronous distributed training. The default
          (`None`) is to use the `tf.get_variable` default.
        weightnorm: An optional (boolean) to activate weightnorm for the mean
        kernel.

    Returns
    ----------
        default_loc_scale_fn: Python `callable` which instantiates `loc`, `scale`
        parameters from args: `dtype, shape, name, trainable, add_variable_fn`.

    """

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates `loc`, `scale` and weightnorm parameters."""
        loc = add_variable_fn(
            name=name + "_loc",
            shape=shape,
            initializer=loc_initializer,
            regularizer=loc_regularizer,
            constraint=loc_constraint,
            dtype=dtype,
            trainable=trainable,
        )
        if weightnorm:
            g = add_variable_fn(
                name=name + "_wn",
                shape=shape,
                initializer=tf.constant_initializer(1.4142),
                constraint=loc_constraint,
                regularizer=loc_regularizer,
                dtype=dtype,
                trainable=trainable,
            )
            loc_wn = tfp_util.DeferredTensor(
                loc, lambda x: (tf.multiply(nn_impl.l2_normalize(x), g))
            )
        # loc = tfp_util.DeferredTensor(loc, lambda x: (nn_impl.l2_normalize(x)))
        if is_singular:
            if weightnorm:
                return loc_wn, None
            else:
                return loc, None

        untransformed_scale = add_variable_fn(
            name=name + "_untransformed_scale",
            shape=shape,
            initializer=untransformed_scale_initializer,
            regularizer=untransformed_scale_regularizer,
            constraint=untransformed_scale_constraint,
            dtype=dtype,
            trainable=trainable,
        )
        scale = tfp_util.DeferredTensor(
            untransformed_scale,
            lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)),
        )
        if weightnorm:
            return loc_wn, scale
        else:
            return loc, scale

    return _fn


def default_mean_field_normal_fn(
    is_singular=False,
    loc_initializer=tf.keras.initializers.he_normal(),
    untransformed_scale_initializer=tf.constant_initializer(0.0001),
    loc_regularizer=None,  # tf.keras.regularizers.l2(), #None
    untransformed_scale_regularizer=None,  # tf.keras.regularizers.l2(), #None
    loc_constraint=None,  # tf.keras.constraints.UnitNorm(axis = [0, 1, 2,3]),
    untransformed_scale_constraint=None,
    weightnorm=False,
):
    """Function for deterministic/variational layers.

    This function produces a closure which produces `tfd.Normal`or
    tfd.Deterministic parameterized by a `loc` and `scale` each created
    using `tf.get_variable`.

    Parameters
    ----------
        is_singular: Python `bool` if `True`, forces the special case limit of
          `scale->0`, i.e., a `Deterministic` distribution. and if set False
          put a variational layer.
        loc_initializer: Initializer function for the `loc` parameters.
          The default is `tf.random_normal_initializer(mean=0., stddev=0.1)`.
        untransformed_scale_initializer: Initializer function for the `scale`
          parameters. Default value: `tf.random_normal_initializer(mean=-3.,
          stddev=0.1)`. This implies the softplus transformed result is initialized
          near `0`. It allows a `Normal` distribution with `scale` parameter set to
          this value to approximately act like a point mass.
        loc_regularizer: Regularizer function for the `loc` parameters.
        untransformed_scale_regularizer: Regularizer function for the `scale`
          parameters.
        loc_constraint: An optional projection function to be applied to the
          loc after being updated by an `Optimizer`. The function must take as input
          the unprojected variable and must return the projected variable (which
          must have the same shape). Constraints are not safe to use when doing
          asynchronous distributed training.
        untransformed_scale_constraint: An optional projection
        function to behttps://github.com/FedericoVasile1/Project8
          applied to the `scale` parameters after being updated by an `Optimizer`
          (e.g. used to implement norm constraints or value constraints). The
          function must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are not
          safe to use when doing asynchronous distributed training.

    Returns
    ----------
        default_mean_field_normal_fn: Python `callable` which instantiates layers.
    """
    loc_scale_fn = default_loc_scale_fn(
        is_singular=is_singular,
        loc_initializer=loc_initializer,
        untransformed_scale_initializer=untransformed_scale_initializer,
        loc_regularizer=loc_regularizer,
        untransformed_scale_regularizer=untransformed_scale_regularizer,
        loc_constraint=loc_constraint,
        untransformed_scale_constraint=untransformed_scale_constraint,
        weightnorm=weightnorm,
    )

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
        if scale is None:
            dist = deterministic_lib.Deterministic(loc=loc)
        else:
            dist = normal_lib.Normal(loc=loc, scale=scale)
        batch_ndims = tf2.size(dist.batch_shape_tensor())
        return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return _fn


def divergence_fn_bayesian(prior_std, examples_per_epoch):
    """Computes KLD loss.

    Computes KLD function for ELBO loss with
    examples per epochs as scaling parameter.

    Parameters
    ----------
        prior_std: int, scale parameter for the priors
        examples_per_epoch: int, scaling number (batchsize)

    Returns
    ----------
        Scaled KLD loss.
    """

    def divergence_fn(q, p, _):
        log_probs = tfd.LogNormal(0.0, prior_std).log_prob(p.stddev())
        out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
        return out / examples_per_epoch

    return divergence_fn


def prior_fn_for_bayesian(init_scale_mean=-1, init_scale_std=0.1):
    """Set priors for the variational layers, possibly trainable.

    Parameters
    ----------
        iniy_scale_mean: int, mean initialized value
        init_scale_std: int, scale initiatlization

    Returns
    ----------
        prior_fn: Python `callable` which instantiates `loc`, `scale`.
    """

    def prior_fn(dtype, shape, name, _, add_variable_fn):
        untransformed_scale = add_variable_fn(
            name=name + "_untransformed_scale",
            shape=(1,),
            initializer=tf.compat.v1.initializers.random_normal(
                mean=init_scale_mean, stddev=init_scale_std
            ),
            dtype=dtype,
            trainable=True,
        )
        loc = add_variable_fn(
            name=name + "_loc",
            initializer=tf.keras.initializers.Zeros(),
            shape=shape,
            dtype=dtype,
            trainable=True,
        )
        scale = 1e-4 + tf.nn.softplus(untransformed_scale)
        dist = tfd.Normal(loc=loc, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn


def normal_prior(prior_std=1.0):
    """Sets normal distributions prior.

    Parameters
    ----------
        prior_std: int, scale parameter, default setting is 1.0.

    Returns
    ----------
        prior_fn: Python `callable`normal distribution.
    """

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        dist = tfd.Normal(
            loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype((prior_std))
        )
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn
