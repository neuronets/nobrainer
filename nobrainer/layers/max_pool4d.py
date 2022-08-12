# import tensorflow as tf
# from tensorflow.python.ops import gen_nn_ops

# def _get_sequence(value, n, channel_index, name):
#   """Formats a value input for gen_nn_ops."""
#   # Performance is fast-pathed for common cases:
#   # `None`, `list`, `tuple` and `int`.
#   if value is None:
#     return [1] * (n + 2)

#   # Always convert `value` to a `list`.
#   if isinstance(value, list):
#     pass
#   elif isinstance(value, tuple):
#     value = list(value)
#   elif isinstance(value, int):
#     value = [value]
#   elif not isinstance(value, collections_abc.Sized):
#     value = [value]
#   else:
#     value = list(value)  # Try casting to a list.

#   len_value = len(value)

#   # Fully specified, including batch and channel dims.
#   if len_value == n + 2:
#     return value

#   # Apply value to spatial dims only.
#   if len_value == 1:
#     value = value * n  # Broadcast to spatial dimensions.
#   elif len_value != n:
#     raise ValueError(f"{name} should be of length 1, {n} or {n + 2}. "
#                      f"Received: {name}={value} of length {len_value}")

#   # Add batch and channel dims (always 1).
#   if channel_index == 1:
#     return [1, 1] + value
#   else:
#     return [1] + value + [1]

# @tf_export("nn.max_pool4d")
# @dispatch.add_dispatch_support
# def max_pool4d(input, ksize, strides, padding, data_format="NVDHWC", name=None):
#     """Performs the max pooling on the input.
#     Args:
#         input: A 6-D `Tensor` of the format specified by `data_format`.
#         ksize: An int or list of `ints` that has length `1`, `3` or `5`. The size of
#         the window for each dimension of the input tensor.
#         strides: An int or list of `ints` that has length `1`, `3` or `5`. The
#           stride of the sliding window for each dimension of the input tensor.
#         padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm. See
#           [here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
#           for more information.
#         data_format: An optional string from: "NVDHWC", "NCVDHW". Defaults to "NVDHWC".
#           The data format of the input and output data. With the default format
#           "NVDHWC", the data is stored in the order of: [batch, in_depth, in_height,
#             in_width, in_channels]. Alternatively, the format could be "NCVDHW", the
#           data storage order is: [batch, in_channels, in_volumes, in_depth, in_height,
#             in_width].
#         name: A name for the operation (optional).
#     Returns:
#         A `Tensor` of format specified by `data_format`.
#         The max pooled output tensor.
#   """
#   with ops.name_scope(name, "MaxPool4D", [input]) as name:
#     if data_format is None:
#       data_format = "NVDHWC"
#     channel_index = 1 if data_format.startswith("NC") else 5

#     ksize = _get_sequence(ksize, 3, channel_index, "ksize")
#     strides = _get_sequence(strides, 3, channel_index, "strides")

#     return gen_nn_ops.max_pool4d(
#         input,
#         ksize=ksize,
#         strides=strides,
#         padding=padding,
#         data_format=data_format,
#         name=name)
