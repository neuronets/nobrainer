import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
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
    loc_initializer=tf1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf1.initializers.random_normal(
        mean=-3.0, stddev=0.1
    ),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None,
    weightnorm=False,
):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates `loc`, `scale` parameters."""
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
    is_singular=True,
    loc_initializer=tf.keras.initializers.he_normal(),
    untransformed_scale_initializer=tf.constant_initializer(0.0001),
    loc_regularizer=None,  # tf.keras.regularizers.l2(), #None
    untransformed_scale_regularizer=None,  # tf.keras.regularizers.l2(), #None
    loc_constraint=None,  # tf.keras.constraints.UnitNorm(axis = [0, 1, 2,3]),
    untransformed_scale_constraint=None,
    weightnorm=False,
):
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
    def divergence_fn(q, p, _):
        log_probs = tfd.LogNormal(0.0, prior_std).log_prob(p.stddev())
        out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
        return out / examples_per_epoch

    return divergence_fn


def prior_fn_for_bayesian(init_scale_mean=-1, init_scale_std=0.1):
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
    """Defines normal distributions prior for Bayesian neural network."""

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        dist = tfd.Normal(
            loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype((prior_std))
        )
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn
