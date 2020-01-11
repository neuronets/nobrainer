"""Custom padding layers for nobrainer."""

import tensorflow as tf
from tensorflow.keras import layers


class ZeroPadding3DChannels(layers.Layer):
    """Pad the last dimension of a 5D tensor symmetrically with zeros.

    This is meant for 3D convolutions, where tensors are 5D.
    """

    def __init__(self, padding, **kwds):
        self.padding = padding
        # batch, x, y, z, channels
        self._paddings = [[0, 0], [0, 0], [0, 0], [0, 0], [self.padding, self.padding]]
        super(ZeroPadding3DChannels, self).__init__(**kwds)

    def call(self, x):
        return tf.pad(x, paddings=self._paddings, mode="CONSTANT", constant_values=0)
