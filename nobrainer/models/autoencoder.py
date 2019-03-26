"""Model definition for 3D Autoencoder.
"""

import tensorflow as tf
from tensorflow.keras import layers
import math

def autoencoder(input_shape, encoding_dim=512, n_base_filters=16, batchnorm=True, batch_size=None, name='autoencoder'):
	"""Instantiate 3D Autoencoder architecture."""

	conv_kwds = {
		'kernel_size': (4, 4, 4),
		'activation': None,
		'padding': 'same',
		# 'kernel_regularizer': tf.keras.regularizers.l2(0.001),
	}

	conv_transpose_kwds = {
		'kernel_size': (4, 4, 4),
		'strides': 2,
		'activation': None,
		'padding': 'same',
		# 'kernel_regularizer': tf.keras.regularizers.l2(0.001),
	}

	n_layers = int(math.log(input_shape[0], 2))

	# Input layer
	inputs = x = layers.Input(shape=input_shape, batch_size=batch_size)

	# Encoder
	for i in range(n_layers):
		n_filters = min(n_base_filters*(2**(i)), encoding_dim)

		x = layers.Conv3D(n_filters, strides=2, **conv_kwds)(x)
		if batchnorm:
			x = layers.BatchNormalization()(x)
		x = layers.ReLU()(x)

	x = layers.Flatten()(x)
	encoding = x = layers.Dense(encoding_dim, activation='tanh')(x)
	
	# Decoder
	x = layers.Reshape((1, 1, 1, encoding_dim))(x)
	for i in range(n_layers):
		n_filters = min(n_base_filters*(2**(n_layers-i-1)), encoding_dim)

		x = layers.Conv3DTranspose(n_filters, **conv_transpose_kwds)(x)
		if batchnorm:
			x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU()(x)

	# Output layer
	outputs = tf.keras.layers.Conv3D(1, 3, activation='tanh', padding='same')(x)

	return tf.keras.models.Model(inputs=inputs, outputs=outputs)