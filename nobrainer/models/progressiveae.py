"""Model definition for ProgressiveAE.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def progressiveae(
    latent_size,
    label_size=0,
    num_channels=1,
    dimensionality=3,
    e_fmap_base=2048,
    d_fmap_base=2048,
    e_fmap_max=512,
    d_fmap_max=256,
):
    """Instantiate ProgressiveAE Architecture.

    Parameters
    ----------
    latent_size: int, size of the latent space to use for generating images
    label_size: int, number of labels for conditional image synthesis (WIP)
    num_channels: int, number of channels in the generated images
    dimensionality: int, one of [2, 3], number of dimensions in the image
    e_fmap_base: int, parameter to determine width of encoder
    d_fmap_base: int, parameter to determine width of decoder
    e_fmap_max: int, parameter to determine width of encoder
    d_fmap_max: int, parameter to determine width of decoder

    Returns
    -------
    Encoder and Decoder
    """
    encoder = Encoder(
        latent_size,
        label_size=label_size,
        num_channels=num_channels,
        dimensionality=dimensionality,
        fmap_base=e_fmap_base,
        fmap_max=e_fmap_max,
    )

    decoder = Decoder(
        latent_size,
        label_size=label_size,
        num_channels=num_channels,
        dimensionality=dimensionality,
        fmap_base=d_fmap_base,
        fmap_max=d_fmap_max,
    )

    return encoder, decoder


class Encoder(tf.keras.Model):
    def __init__(
        self,
        latent_size,
        label_size=0,
        num_channels=1,
        fmap_base=2048,
        fmap_max=512,
        dimensionality=3,
    ):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        self.label_size = label_size

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max

        self.num_channels = num_channels
        self.dimensionality = dimensionality

        self.current_resolution = 2
        self.current_width = 2**self.current_resolution

        self.Conv = getattr(layers, "Conv{}D".format(self.dimensionality))
        self.Upsampling = getattr(layers, "UpSampling{}D".format(self.dimensionality))
        self.AveragePooling = getattr(
            layers, "AveragePooling{}D".format(self.dimensionality)
        )

        self.highest_resolution_block = self.make_Eblock(
            (self._nf(self.current_resolution)),
            name=("e_block_{}".format(self.current_resolution)),
        )
        self.resolution_blocks = []

        self.Base_Dense = layers.Dense(self.latent_size)
        self.Head_Conv = self.Conv(
            (self._nf(self.current_resolution)),
            kernel_size=1,
            padding="same",
        )
        self.Base_Conv = self.Conv(
            (self._nf(self.current_resolution - 1)),
            kernel_size=1,
            padding="same",
        )

        images_shape = (
            (None,)
            + (int(2.0**self.current_resolution),) * self.dimensionality
            + (self.num_channels,)
        )
        alpha_shape = (1,)

        self.build([images_shape, alpha_shape])

    def _pixel_norm(self, epsilon=1e-08):
        """
        Pixelwise normalization
        """
        return layers.Lambda(
            lambda x: x
            * tf.math.rsqrt(
                tf.reduce_mean((tf.square(x)), axis=(-1), keepdims=True) + epsilon
            )
        )

    def _weighted_sum(self):
        """
        Weighted interpolation helper for fading in new layers as its progressively added
        """
        return layers.Lambda(
            lambda inputs: (1 - inputs[2]) * inputs[0] + (inputs[2]) * inputs[1]
        )

    def _nf(self, stage):
        """
        Computes number of filters for a conv layer
        """
        return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

    def update_res(self):
        """
        Updates the current resolution of the model
        """
        self.current_resolution += 1
        self.current_width = 2**self.current_resolution

    def make_Ehead(self, x):
        """
        Creates the end of the encoder. The output node gives the latent representation.
        """
        x = layers.Flatten()(x)
        x = self.Base_Dense(x)

        return x

    def make_Eblock(self, nf, kernel_size=4, name="eblock", input_shape=None):
        """
        Creates an encoder block
        """
        block_layers = []
        block_layers.append(
            self.Conv(nf, kernel_size=kernel_size, strides=2, padding="same")
        )
        block_layers.append(layers.Activation(tf.nn.leaky_relu))
        return models.Sequential(block_layers, name=name)

    def make_Ebase(self, x, y, alpha):
        """
        Creates an encoder base. Performs the fading using the alpha factor
        """

        x = self.AveragePooling()(x)
        x = self.Base_Conv(x)
        x = self._weighted_sum()([x, y, alpha])
        return x

    def add_resolution(self):
        """
        Adds a resolution step to the trainable encoder by creating a
        encoder block and inserting at the start of the existing encoder
        to increase the resolution of the images encoded by a factor of 2.
        The new layers are faded in and controlled by the alpha parameter.
        """

        self.update_res()

        self.resolution_blocks.append(self.highest_resolution_block)
        self.highest_resolution_block = self.make_Eblock(
            self._nf(self.current_resolution - 1),
            name="eblock{}".format(self.current_resolution),
        )

        self.Base_Conv = self.Conv(
            filters=self._nf(self.current_resolution - 1),
            kernel_size=1,
            padding="same",
        )
        self.Head_Conv = self.Conv(
            filters=self._nf(self.current_resolution), kernel_size=1, padding="same"
        )

        images_shape = (
            (None,)
            + (int(2.0**self.current_resolution),) * self.dimensionality
            + (self.num_channels,)
        )
        alpha_shape = (1,)

        self.build([images_shape, alpha_shape])

    def call(self, inputs):
        images, alpha = inputs

        x = self.Head_Conv(images)
        y = self.highest_resolution_block(x)
        x = self.make_Ebase(images, y, alpha)

        for e_block in self.resolution_blocks[::-1]:
            x = e_block(x)

        return self.make_Ehead(x)


class Decoder(tf.keras.Model):
    def __init__(
        self,
        latent_size,
        label_size=0,
        num_channels=1,
        fmap_base=2048,
        fmap_max=256,
        dimensionality=3,
    ):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.label_size = label_size

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max

        self.num_channels = num_channels
        self.dimensionality = dimensionality

        self.Conv = getattr(layers, "Conv{}D".format(self.dimensionality))
        self.Upsampling = getattr(layers, "UpSampling{}D".format(self.dimensionality))

        self.current_resolution = 2
        self.current_width = 2**self.current_resolution

        self.highest_resolution_block = self.make_Dblock(
            self._nf(self.current_resolution),
            name="d_block_{}".format(self.current_resolution),
        )
        self.resolution_blocks = []

        self.Head_Conv1 = self.Conv(filters=(self.num_channels), kernel_size=1)
        self.Head_Conv2 = self.Conv(filters=(self.num_channels), kernel_size=1)
        self.Base_Dense = tf.keras.layers.Dense(
            units=self._nf(1) * 2**self.dimensionality
        )

        self.build([(None, latent_size), (1,)])

    def update_res(self):
        self.current_resolution += 1
        self.current_width = 2**self.current_resolution

    def _pixel_norm(self, epsilon=1e-08):
        """
        Pixelwise normalization
        """
        return layers.Lambda(
            lambda x: x
            * tf.math.rsqrt(
                tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon
            )
        )

    def _weighted_sum(self):
        """
        Weighted interpolation helper for fading in new layers as its progressively added
        """
        return layers.Lambda(
            lambda inputs: (1 - inputs[2]) * inputs[0] + (inputs[2]) * inputs[1]
        )

    def _nf(self, stage):
        """
        Computes number of filters for a conv layer
        """
        return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

    def make_Dbase(self, latents):
        """
        Creates the base of the decoder with inputs of size latent_size
        and the alpha value to determine level of interpolation/fading
        """

        x = self._pixel_norm()(latents)
        x = self.Base_Dense(x)
        x = layers.Reshape([2] * self.dimensionality + [self._nf(1)])(x)

        return x

    def make_Dblock(self, nf, kernel_size=4, name=""):
        """
        Creates a decoder block
        """

        block_layers = []

        block_layers.append(self.Upsampling())
        block_layers.append(
            self.Conv(nf, kernel_size=kernel_size, strides=1, padding="same")
        )
        block_layers.append(layers.Activation(tf.nn.leaky_relu))

        block_layers.append(self._pixel_norm())
        block_layers.append(
            self.Conv(nf, kernel_size=kernel_size, strides=1, padding="same")
        )
        block_layers.append(layers.Activation(tf.nn.leaky_relu))

        block_layers.append(self._pixel_norm())

        return models.Sequential(block_layers, name=name)

    def make_Dhead(self, x, y, alpha):
        """
        Creates a decoder head. Performs the fading/interpolation using the alpha parameter
        """

        x = self.Upsampling()(x)
        x = self.Head_Conv1(x)

        y = self.Head_Conv2(y)

        x = self._weighted_sum()([x, y, alpha])

        x = tf.keras.layers.Activation("tanh")(x)

        return x

    def add_resolution(self):
        """
        Adds a resolution step to the trainable decoder by creating a
        decoder block and inserting at the start of the existing decoder
        to increase the resolution of the images decoder by a factor of 2.
        The new layers are faded in and controlled by the alpha parameter.
        """

        self.update_res()

        self.resolution_blocks.append(self.highest_resolution_block)
        self.highest_resolution_block = self.make_Dblock(
            self._nf(self.current_resolution),
            name="d_block_{}".format(self.current_resolution),
        )

        self.Head_Conv1 = self.Conv(filters=(self.num_channels), kernel_size=1)
        self.Head_Conv2 = self.Conv(filters=(self.num_channels), kernel_size=1)

        self.build([(None, self.latent_size), (1,)])

    def call(self, inputs):
        latents, alpha = inputs

        x = self.make_Dbase(latents)

        for d_block in self.resolution_blocks:
            x = d_block(x)

        y = self.highest_resolution_block(x)
        return self.make_Dhead(x, y, alpha)
