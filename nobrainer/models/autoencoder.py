"""Model definition for 3D Autoencoder.
"""

import tensorflow as tf
from tensorflow.keras import layers

def autoencoder(input_shape, n_layers=3, n_base_filters=16, activation='relu', batchnorm=False, batch_size=None, name='unet'):
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

	inputs = x = layers.Input(shape=input_shape, batch_size=batch_size)

	# Encoder

	for i in range(n_layers):
		n_filters = n_base_filters*(2**i)

		x = layers.Conv3D(n_filters, strides=1, **conv_kwds)(x)
		if batchnorm:
			x = layers.BatchNormalization()(x)
		x = layers.Activation(activation)(x)

		x = layers.Conv3D(n_filters, strides=2, **conv_kwds)(x)
		if batchnorm:
			x = layers.BatchNormalization()(x)
		x = layers.Activation(activation)(x)

	# Decoder

	for i in range(n_layers):
		n_filters = n_base_filters*(2**(n_layers-i))

		x = layers.Conv3DTranspose(n_filters, strides=2, **conv_kwds)(x)
		if batchnorm:
			x = layers.BatchNormalization()(x)
		x = layers.Activation(activation)(x)

	# Output layer

	pred = tf.keras.layers.Conv3D(1, 3, activation='tanh', padding='same')(x)

	return tf.keras.models.Model(inputs=inputs, outputs=pred)