# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras variational convolution layers, which also incorporate weight
noramlization. The acronym VWN stands for variational weight normalization.

The layers in this file learn distributions of weights (i.e., the mean and
variance of Gaussian distributions).

This file was modified from the standard `tf.keras.layers.convolutional`
module. The original version of this file can be found at
https://github.com/tensorflow/tensorflow/blob/507ffd071f0a2af8a9f5678dae225e50479b44c9/tensorflow/python/keras/layers/convolutional.py
"""

import math

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


class VWNConv(Layer):
  """Abstract nD variational convolution layer that implements weight
  normalization (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    TODO: if is_mc is true, then variational dropout is applied.
    is_mc: Boolean, whether the layer is Monte Carlo style. If true, this layer
        learns a normal distribution, from which weights can be sampled.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               activation=None,
               is_mc=True,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(VWNConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and ``SeparableConv1D`.')
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.is_mc = tf.cast(is_mc, dtype=tf.bool)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)
    self.a_initializer = initializers.Constant(1e-04)  # ADDED  (what is a)  (use keras initializers??)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    # ADDED
    g_shape = [1 for _ in self.kernel_size] + [1, self.filters]

    # self.kernel = self.add_weight(
    #     name='kernel',
    #     shape=kernel_shape,
    #     initializer=self.kernel_initializer,
    #     regularizer=self.kernel_regularizer,
    #     constraint=self.kernel_constraint,
    #     trainable=True,
    #     dtype=self.dtype)

    self.v = self.add_weight(
        name='v',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=None,
        constraint=None,
        trainable=True,
        dtype=self.dtype)
    # tf.summary.histogram(self.v.name, self.v)
    self.g = self.add_weight(
        name='g',
        shape=g_shape,
        initializer=tf.constant_initializer(math.sqrt(2)),
        regularizer=None,
        constraint=None,
        trainable=True,
        dtype=self.dtype)
    tf.summary.histogram(self.g.name, self.g)
    self.v_norm = nn.l2_normalize(
        self.v, [i for i in range(len(self.kernel_size) + 1)])

    self.kernel_m = tf.multiply(self.g, self.v_norm, name='kernel_m')
    tf.summary.histogram(self.kernel_m.name, self.kernel_m)
    self.kernel_a = self.add_weight(
        name='kernel_a',
        shape=kernel_shape,
        initializer=self.a_initializer,
        regularizer=None,
        constraint=None,
        trainable=True,
        dtype=self.dtype)
    tf.summary.histogram(self.kernel_a.name, self.kernel_a)
    self.kernel_sigma = tf.abs(self.kernel_a, name='kernel_sigma')
    tf.summary.histogram(self.kernel_sigma.name, self.kernel_sigma)
    tf.summary.scalar(self.kernel_sigma.name, tf.reduce_mean(self.kernel_sigma))
    self.kernel = self.kernel_m

    if self.use_bias:
        self.bias_m = self.add_weight(
            name='bias_m',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
        tf.summary.histogram(self.bias_m.name, self.bias_m)
        self.bias_a = self.add_weight(
            name='bias_a',
            shape=(self.filters,),
            initializer=self.a_initializer,
            regularizer=None,
            constraint=None,
            trainable=True,
            dtype=self.dtype)
        tf.summary.histogram(self.bias_a.name, self.bias_a)
        self.bias_sigma = tf.abs(self.bias_a, name='bias_sigma')
        tf.summary.histogram(self.bias_sigma.name, self.bias_sigma)
        # tf.add_to_collection('sigmas',self.bias_sigma)
        tf.summary.scalar(self.bias_sigma.name, tf.reduce_mean(self.bias_sigma))
        self.bias = self.bias_m
      # self.bias = self.add_weight(
      #     name='bias',
      #     shape=(self.filters,),
      #     initializer=self.bias_initializer,
      #     regularizer=self.bias_regularizer,
      #     constraint=self.bias_constraint,
      #     trainable=True,
      #     dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.get_shape(),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=op_padding,
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2))
    self.built = True

  def call(self, inputs):
    # outputs = self._convolution_op(inputs, self.kernel)
    # This performs two convolution operations
    outputs_mean = self._convolution_op(inputs, self.kernel_m)
    outputs_var = self._convolution_op(tf.square(inputs), tf.square(self.kernel_sigma))

    if self.use_bias:
      if self.data_format == 'channels_first':
        # TODO: test the channels first implementation.
        raise NotImplementedError(
            "The data format 'channels_first' is not supported yet.")
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias_m = array_ops.reshape(self.bias_m, (1, self.filters, 1))
          bias_sigma = array_ops.reshape(self.bias_sigma, (1, self.filters, 1))
          outputs_mean += bias_m
          outputs_var += tf.square(bias_sigma)
        else:
          # TODO: we might have to reshape the outputs before bias_add.
          # outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
          outputs_mean = nn.bias_add(outputs_mean, self.bias_m, data_format='NCHW')
          outputs_var = nn.bias_add(outputs_var, tf.square(self.bias_sigma), data_format='NCHW')
      else:
        # outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')
        outputs_mean = nn.bias_add(outputs_mean, self.bias_m, data_format='NHWC')
        outputs_var = nn.bias_add(outputs_var, tf.square(self.bias_sigma), data_format='NHWC')

    def apply_variational_approximation():
      # The value rand allows us to sample a random value from our learned
      # distribution.
      rand = tf.random_normal(shape=tf.shape(self.g), dtype=self.dtype)
      return outputs_mean + tf.sqrt(outputs_var + backend.epsilon()) * rand

    # If is_mc is true, we sample a weight from the learned gaussian
    # distribution. If not, we choose the most likely weight (i.e., the mean
    # of the distribution). A side effect of this, though, is that we always
    # have to do two convolutional operations (one for mean and the other for
    # variance).
    outputs = tf.cond(
        self.is_mc,
        true_fn=apply_variational_approximation,
        false_fn=lambda: outputs_mean)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        # TODO (kaczmarj): add is_mc and a_initializer.
    }
    base_config = super(VWConv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding


class VWNConv1D(VWNConv):
  """1D variational convolution layer that implements weight normalization
  (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  with the layer input over a single spatial (or temporal) dimension
  to produce a tensor of outputs.
  If `use_bias` is True, a bias vector is created and added to the outputs.
  Finally, if `activation` is not `None`,
  it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide an `input_shape` argument
  (tuple of integers or `None`, e.g.
  `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
  or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
          specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
          specifying the stride length of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
          `"causal"` results in causal (dilated) convolutions, e.g. output[t]
          does not depend on input[t+1:]. Useful when modeling temporal data
          where the model should not violate the temporal order.
          See [WaveNet: A Generative Model for Raw Audio, section
            2.1](https://arxiv.org/abs/1609.03499).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
      dilation_rate: an integer or tuple/list of a single integer, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      is_mc: Boolean, whether the layer is Monte Carlo style. If true, this
          layer learns a normal distribution, from which weights can be sampled.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`

  Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)`
      `steps` value might have changed due to padding or strides.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               is_mc=True,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(VWNConv1D, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        is_mc=is_mc,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    if self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
    return super(VWNConv1D, self).call(inputs)


class VWNConv2D(VWNConv):
  """2D variational convolution layer that implements weight normalization
  (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          height and width of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the height and width.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      is_mc: Boolean, whether the layer is Monte Carlo style. If true, this
          layer learns a normal distribution, from which weights can be sampled.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               is_mc=True,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(VWNConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        is_mc=is_mc,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


class VWNConv3D(VWNConv):
  """3D variational convolution layer that implements weight normalization
  (e.g. spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
  with a single channel,
  in `data_format="channels_last"`.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 3 integers, specifying the
          depth, height and width of the 3D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 3 integers,
          specifying the strides of the convolution along each spatial
            dimension.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 3 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      is_mc: Boolean, whether the layer is Monte Carlo style. If true, this
          layer learns a normal distribution, from which weights can be sampled.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      5D tensor with shape:
      `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
        data_format='channels_first'
      or 5D tensor with shape:
      `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
        data_format='channels_last'.

  Output shape:
      5D tensor with shape:
      `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
        data_format='channels_first'
      or 5D tensor with shape:
      `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
        data_format='channels_last'.
      `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
        changed due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
               activation=None,
               is_mc=True,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(VWNConv3D, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        is_mc=is_mc,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


# Aliases
VariationalWeightNormConvolution1D = VWNConv1D
VariationalWeightNormConvolution2D = VWNConv2D
VariationalWeightNormConvolution3D = VWNConv3D
