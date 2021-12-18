"""Convolutional variational layers."""

from __future__ import absolute_import, division, print_function

import tensorflow.compat.v2 as tf
from tensorflow.python.layers import utils as tf_layers_util
from tensorflow.python.ops import nn_ops
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.layers import util as tfp_layers_util

__all__ = [
    "Convolution3DReparameterization",
]
doc_args = """activation: Activation function. Set it to None to maintain a
      linear activation.
  activity_regularizer: Regularizer function for the output.
  kernel_posterior_fn: Python `callable` which creates
    `tfd.Distribution` instance representing the surrogate
    posterior of the `kernel` parameter. Default value:
    `default_mean_field_normal_fn()`.
  kernel_posterior_tensor_fn: Python `callable` which takes a
    `tfd.Distribution` instance and returns a representative
    value. Default value: `lambda d: d.sample()`.
  kernel_prior_fn: Python `callable` which creates `tfd`
    instance. See `default_mean_field_normal_fn` docstring for required
    parameter signature.
    Default value: `tfd.Normal(loc=0., scale=1.)`.
  kernel_divergence_fn: Python `callable` which takes the surrogate posterior
    distribution, prior distribution and random variate sample(s) from the
    surrogate posterior and computes or approximates the KL divergence. The
    distributions are `tfd.Distribution`-like instances and the
    sample is a `Tensor`.
  bias_posterior_fn: Python `callable` which creates
    `tfd.Distribution` instance representing the surrogate
    posterior of the `bias` parameter. Default value:
    `default_mean_field_normal_fn(is_singular=True)` (which creates an
    instance of `tfd.Deterministic`).
  bias_posterior_tensor_fn: Python `callable` which takes a
    `tfd.Distribution` instance and returns a representative
    value. Default value: `lambda d: d.sample()`.
  bias_prior_fn: Python `callable` which creates `tfd` instance.
    See `default_mean_field_normal_fn` docstring for required parameter
    signature. Default value: `None` (no prior, no variational inference)
  bias_divergence_fn: Python `callable` which takes the surrogate posterior
    distribution, prior distribution and random variate sample(s) from the
    surrogate posterior and computes or approximates the KL divergence. The
    distributions are `tfd.Distribution`-like instances and the
    sample is a `Tensor`."""


"""Variational convolutional layer Abstract nD convolution layer (private,
  used as implementation base).
  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. It may also include a bias addition and activation function
  on the outputs. It assumes the `kernel` and/or `bias` are drawn from
  distributions.
  By default, the layer implements a stochastic forward pass via
  sampling from the kernel and bias posteriors,
  ```none
  outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
  ```
  where f denotes the layer's calculation.
  The arguments permit separate specification of the surrogate posterior
  (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
  distributions.
"""


class _ConvVariational(tf.keras.layers.Layer):
    @docstring_util.expand_docstring(args=doc_args)
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        is_mc,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        activation=None,
        activity_regularizer=None,
        kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
        kernel_posterior_tensor_fn=lambda d: d.sample(),
        kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
        kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
        bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
            is_singular=True
        ),
        bias_posterior_tensor_fn=lambda d: d.sample(),
        bias_prior_fn=None,
        bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
        **kwargs
    ):
        super(_ConvVariational, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )
        self.rank = rank
        self.is_mc = is_mc
        self.filters = filters
        self.kernel_size = tf_layers_util.normalize_tuple(
            kernel_size, rank, "kernel_size"
        )
        self.strides = tf_layers_util.normalize_tuple(strides, rank, "strides")
        self.padding = tf_layers_util.normalize_padding(padding)
        self.data_format = tf_layers_util.normalize_data_format(data_format)
        self.dilation_rate = tf_layers_util.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.activation = tf.keras.activations.get(activation)
        self.input_spec = tf.keras.layers.InputSpec(ndim=self.rank + 2)
        self.kernel_posterior_fn = kernel_posterior_fn
        self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
        self.kernel_prior_fn = kernel_prior_fn
        self.kernel_divergence_fn = kernel_divergence_fn
        self.bias_posterior_fn = bias_posterior_fn
        self.bias_posterior_tensor_fn = bias_posterior_tensor_fn
        self.bias_prior_fn = bias_prior_fn
        self.bias_divergence_fn = bias_divergence_fn

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = tf.compat.dimension_value(input_shape[channel_axis])
        if input_dim is None:
            raise ValueError("The channel dimension of inputs Found `None`.")
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        # Must have a posterior kernel.
        self.kernel_posterior = self.kernel_posterior_fn(
            dtype, kernel_shape, "kernel_posterior", self.trainable, self.add_variable
        )
        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(
                dtype, kernel_shape, "kernel_prior", self.trainable, self.add_variable
            )
        if self.bias_posterior_fn is None:
            self.bias_posterior = None
        else:
            self.bias_posterior = self.bias_posterior_fn(
                dtype,
                (self.filters,),
                "bias_posterior",
                self.trainable,
                self.add_variable,
            )
        if self.bias_prior_fn is None:
            self.bias_prior = None
        else:
            self.bias_prior = self.bias_prior_fn(
                dtype, (self.filters,), "bias_prior", self.trainable, self.add_variable
            )
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim}
        )
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=tf.TensorShape(kernel_shape),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=tf_layers_util.convert_data_format(
                self.data_format, self.rank + 2
            ),
        )
        self.built = True

    def call(self, inputs):
        inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)
        outputs = self._apply_variational_kernel(inputs)
        outputs = self._apply_variational_bias(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        self._apply_divergence(
            self.kernel_divergence_fn,
            self.kernel_posterior,
            self.kernel_prior,
            self.kernel_posterior_tensor,
            name="divergence_kernel",
        )
        self._apply_divergence(
            self.bias_divergence_fn,
            self.bias_posterior,
            self.bias_prior,
            self.bias_posterior_tensor,
            name="divergence_bias",
        )
        return outputs

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
        Args:
          input_shape: Shape tuple (tuple of integers) or list of shape tuples
            (one per output tensor of the layer). Shape tuples can include None for
            free dimensions, instead of an integer.
        Returns:
          output_shape: A tuple representing the output shape.
        """
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = tf_layers_util.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = tf_layers_util.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters] + new_space)

    def get_config(self):
        """Returns the config of the layer.
        A layer config is a Python dictionary (serializable) containing the
        configuration of a layer. The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        Returns:
          config: A Python dictionary of class keyword arguments and their
            serialized values.
        """
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": (
                tf.keras.activations.serialize(self.activation)
                if self.activation
                else None
            ),
            "activity_regularizer": tf.keras.initializers.serialize(
                self.activity_regularizer
            ),
        }
        function_keys = [
            "kernel_posterior_fn",
            "kernel_posterior_tensor_fn",
            "kernel_prior_fn",
            "kernel_divergence_fn",
            "bias_posterior_fn",
            "bias_posterior_tensor_fn",
            "bias_prior_fn",
            "bias_divergence_fn",
        ]
        for function_key in function_keys:
            function = getattr(self, function_key)
            if function is None:
                function_name = None
                function_type = None
            else:
                function_name, function_type = tfp_layers_util.serialize_function(
                    function
                )
            config[function_key] = function_name
            config[function_key + "_type"] = function_type
        base_config = super(_ConvVariational, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.
        This method is the reverse of `get_config`, capable of instantiating the
        same layer from the config dictionary.
        Args:
          config: A Python dictionary, typically the output of `get_config`.
        Returns:
          layer: A layer instance.
        """
        config = config.copy()
        function_keys = [
            "kernel_posterior_fn",
            "kernel_posterior_tensor_fn",
            "kernel_prior_fn",
            "kernel_divergence_fn",
            "bias_posterior_fn",
            "bias_posterior_tensor_fn",
            "bias_prior_fn",
            "bias_divergence_fn",
        ]
        for function_key in function_keys:
            serial = config[function_key]
            function_type = config.pop(function_key + "_type")
            if serial is not None:
                config[function_key] = tfp_layers_util.deserialize_function(
                    serial, function_type=function_type
                )
        return cls(**config)

    def _apply_variational_bias(self, inputs):
        if self.bias_posterior is None:
            self.bias_posterior_tensor = None
            return inputs
        self.bias_posterior_tensor = self.bias_posterior_tensor_fn(self.bias_posterior)
        outputs = inputs
        if self.data_format == "channels_first":
            if self.rank == 1:
                bias = tf.reshape(self.bias_posterior_tensor, [1, self.filters, 1])
                outputs += bias
            if self.rank == 2:
                outputs = tf.nn.bias_add(
                    outputs, self.bias_posterior_tensor, data_format="NCHW"
                )
            if self.rank == 3:
                outputs_shape = outputs.shape.as_list()
                outputs_4d = tf.reshape(
                    outputs,
                    [
                        outputs_shape[0],
                        outputs_shape[1],
                        outputs_shape[2] * outputs_shape[3],
                        outputs_shape[4],
                    ],
                )
                outputs_4d = tf.nn.bias_add(
                    outputs_4d, self.bias_posterior_tensor, data_format="NCHW"
                )
                outputs = tf.reshape(outputs_4d, outputs_shape)
        else:
            outputs = tf.nn.bias_add(
                outputs, self.bias_posterior_tensor, data_format="NHWC"
            )
        return outputs

    def _apply_divergence(
        self, divergence_fn, posterior, prior, posterior_tensor, name
    ):
        if divergence_fn is None or posterior is None or prior is None:
            divergence = None
            return
        divergence = tf.identity(
            divergence_fn(posterior, prior, posterior_tensor), name=name
        )
        self.add_loss(divergence)


class _ConvReparameterization(_ConvVariational):
    """Abstract nD convolution layer (private, used as implementation base).
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. It may also include a bias addition and activation function
    on the outputs. It assumes the `kernel` and/or `bias` are drawn from
    distributions.
    By default, the layer implements a stochastic forward pass via
    sampling from the kernel and bias posteriors,
    ```none
    outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
    ```
    where f denotes the layer's calculation. It uses the reparameterization
    estimator [(Kingma and Welling, 2014)][1], which performs a Monte Carlo
    approximation of the distribution integrating over the `kernel` and `bias`.
    The arguments permit separate specification of the surrogate posterior
    (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
    distributions.
    #### References
    [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
         _International Conference on Learning Representations_, 2014.
         https://arxiv.org/abs/1312.6114
    """

    @docstring_util.expand_docstring(args=doc_args)
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        is_mc,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        activation=None,
        activity_regularizer=None,
        kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
        kernel_posterior_tensor_fn=lambda d: d.sample(),
        kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
        kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
        bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
            is_singular=True
        ),
        bias_posterior_tensor_fn=lambda d: d.sample(),
        bias_prior_fn=None,
        bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
        **kwargs
    ):
        super(_ConvReparameterization, self).__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            is_mc=is_mc,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=tf.keras.activations.get(activation),
            activity_regularizer=activity_regularizer,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
            kernel_prior_fn=kernel_prior_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_posterior_fn=bias_posterior_fn,
            bias_posterior_tensor_fn=bias_posterior_tensor_fn,
            bias_prior_fn=bias_prior_fn,
            bias_divergence_fn=bias_divergence_fn,
            **kwargs
        )

    def _apply_variational_kernel(self, inputs):
        if not isinstance(
            self.kernel_posterior, independent_lib.Independent
        ) or not isinstance(self.kernel_posterior.distribution, normal_lib.Normal):
            self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
                self.kernel_posterior
            )
            self.kernel_posterior_affine = None
            self.kernel_posterior_affine_tensor = None
            outputs = self._convolution_op(inputs, self.kernel_posterior_tensor)
            return outputs
        else:
            self.kernel_posterior_affine = normal_lib.Normal(
                loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
                scale=self.kernel_posterior.distribution.scale,
            )
            self.kernel_posterior_affine_tensor = self.kernel_posterior_tensor_fn(
                self.kernel_posterior_affine
            )
            self.kernel_posterior_tensor = None
            outputs_m = self._convolution_op(
                inputs, self.kernel_posterior.distribution.loc
            )
            outputs_v = self._convolution_op(
                tf.square(inputs),
                tf.square(self.kernel_posterior.distribution.stddev()),
            )
            k_size = tf_layers_util.normalize_tuple(self.kernel_size, 3, "k_size")
            g_shape = [1 for i in k_size] + [1, self.filters]
            outputs_e = tf.random.normal(shape=g_shape, dtype=self.dtype)
            if self.is_mc:
                err = tf.sqrt(tf.add(outputs_v, tf.keras.backend.epsilon())) * outputs_e
                return outputs_m + err
            else:
                return outputs_m


class Conv3DReparameterization(_ConvReparameterization):
    """3D convolution layer (e.g. spatial convolution over volumes).
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. It may also include a bias addition and activation function
    on the outputs. It assumes the `kernel` and/or `bias` are drawn from
    distributions.
    By default, the layer implements a stochastic forward pass via
    sampling from the kernel and bias posteriors,
    ```none
    outputs = f(inputs; kernel, bias), kernel, bias ~ posterior
    ```
    where f denotes the layer's calculation. It uses the reparameterization
    estimator [(Kingma and Welling, 2014)][1], which performs a Monte Carlo
    approximation of the distribution integrating over the `kernel` and `bias`.
    The arguments permit separate specification of the surrogate posterior
    (`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
    distributions.
    Upon being built, this layer adds losses (accessible via the `losses`
    property) representing the divergences of `kernel` and/or `bias` surrogate
    posteriors and their respective priors. When doing minibatch stochastic
    optimization, make sure to scale this loss such that it is applied just once
    per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
    you should pass `kl / num_examples_per_epoch` to your optimizer).
    #### Examples
    We illustrate a Bayesian neural network with [variational inference](
    https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
    assuming a dataset of `features` and `labels`.
    ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape([256, 32, 32, 3]),
        tfp.layers.Convolution3DReparameterization(
            64, kernel_size=5, padding='SAME', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling3D(pool_size=[2, 2, 2],
                                     strides=[2, 2, 2],
                                     padding='SAME'),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseReparameterization(10),
    ])
    logits = model(features)
    neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    kl = sum(model.losses)
    loss = neg_log_likelihood + kl
    train_op = tf.train.AdamOptimizer().minimize(loss)
    ```
    It uses reparameterization gradients to minimize the
    Kullback-Leibler divergence up to a constant, also known as the
    negative Evidence Lower Bound. It consists of the sum of two terms:
    the expected negative log-likelihood, which we approximate via
    Monte Carlo; and the KL divergence, which is added via regularizer
    terms which are arguments to the layer.
    #### References
    [1]: Diederik Kingma and Max Welling. Auto-Encoding Variational Bayes. In
         _International Conference on Learning Representations_, 2014.
         https://arxiv.org/abs/1312.6114
    """

    @docstring_util.expand_docstring(args=doc_args)
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        data_format="channels_last",
        dilation_rate=(1, 1, 1),
        activation=None,
        activity_regularizer=None,
        is_mc=tf.constant(False, dtype=tf.bool),
        kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
        kernel_posterior_tensor_fn=lambda d: d.sample(),
        kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
        kernel_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
        bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
            is_singular=True
        ),
        bias_posterior_tensor_fn=lambda d: d.sample(),
        bias_prior_fn=None,
        bias_divergence_fn=lambda q, p, ignore: kl_lib.kl_divergence(q, p),
        **kwargs
    ):
        # pylint: disable=g-doc-args
        """Construct layer.
        Args:
          filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
          kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, height and width of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
          strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth,
            height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
          padding: One of `"valid"` or `"same"` (case-insensitive).
          data_format: A string, one of `channels_last` (default) or
            `channels_first`. The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape `(batch, depth,
            height, width, channels)` while `channels_first` corresponds to inputs
            with shape `(batch, channels, depth, height, width)`.
          dilation_rate: An integer or tuple/list of 3 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
          ${args}"""
        super(Conv3DReparameterization, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            is_mc=is_mc,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=tf.keras.activations.get(activation),
            activity_regularizer=activity_regularizer,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
            kernel_prior_fn=kernel_prior_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_posterior_fn=bias_posterior_fn,
            bias_posterior_tensor_fn=bias_posterior_tensor_fn,
            bias_prior_fn=bias_prior_fn,
            bias_divergence_fn=bias_divergence_fn,
            **kwargs
        )


Convolution3DReparameterization = Conv3DReparameterization
