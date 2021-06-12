"""Model definition for ProgressiveGAN.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from ..volume import adjust_dynamic_range as _adjust_dynamic_range


def progressivegan(
    latent_size,
    label_size=0,
    num_channels=1,
    dimensionality=3,
    g_fmap_base=8192,
    d_fmap_base=8192,
    g_fmap_max=256,
    d_fmap_max=256,
):
    """Instantiate ProgressiveGAN Architecture.

    Parameters
    ----------
    latent_size: int, size of the latent space to use for generating images
    label_size: int, number of labels for conditional image synthesis (WIP)
    num_channels: int, number of channels in the generated images
    dimensionality: int, one of [2, 3], number of dimensions in the image
    g_fmap_base: int, parameter to determine width of generator
    d_fmap_base: int, parameter to determine width of discriminator
    g_fmap_max: int, parameter to determine width of generator
    d_fmap_max: int, parameter to determine width of discriminator
    name: str, name to give to the resulting model object.

    Returns
    -------
    Generator and Discriminator
    """

    generator = Generator(
        latent_size,
        label_size=label_size,
        num_channels=num_channels,
        dimensionality=dimensionality,
        fmap_base=g_fmap_base,
        fmap_max=g_fmap_max,
    )

    discriminator = Discriminator(
        label_size=label_size,
        num_channels=num_channels,
        dimensionality=dimensionality,
        fmap_base=d_fmap_base,
        fmap_max=d_fmap_max,
    )

    return generator, discriminator


class Generator(tf.keras.Model):
    def __init__(
        self,
        latent_size,
        label_size=0,
        num_channels=1,
        fmap_base=8192,
        fmap_max=256,
        dimensionality=3,
    ):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.label_size = label_size

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.num_channels = num_channels
        self.dimensionality = dimensionality

        self.Conv = getattr(layers, "Conv{}D".format(self.dimensionality))
        self.ConvTranspose = getattr(
            layers, "Conv{}DTranspose".format(self.dimensionality)
        )
        self.Upsampling = getattr(layers, "UpSampling{}D".format(self.dimensionality))

        self.current_resolution = 2
        self.highest_resolution_block = self._make_generator_block(
            self._nf(self.current_resolution),
            name="g_block_{}".format(self.current_resolution),
        )

        self.resolution_blocks = []

        self.base_dense = tf.keras.layers.Dense(
            units=self._nf(1) * (2 ** self.dimensionality)
        )
        self.HeadConv1 = self.Conv(filters=self.num_channels, kernel_size=1)
        self.HeadConv2 = self.Conv(filters=self.num_channels, kernel_size=1)

        self.build([(None, latent_size), (1,)])

    def _pixel_norm(self, epsilon=1e-8):
        """
        Pixelwise normalization
        """
        return layers.Lambda(
            lambda x: x
            * tf.math.rsqrt(
                tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon
            )
        )
        # return layers.BatchNormalization(axis=-1)

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

    def generator_base(self, latents):
        """
        Creates the base of the generator with inputs of size latent_size+label_size
        and the alpha value to determine level of interpolation
        """

        # Latents stage
        x = self._pixel_norm()(latents)
        x = self.base_dense(x)
        x = layers.Reshape([2] * self.dimensionality + [self._nf(1)])(x)

        return x

    def _make_generator_block(self, nf, kernel_size=4, name=""):
        """
        Creates a generator block
        """

        block_layers = []

        # block_layers.append(self.ConvTranspose(nf, kernel_size=3,
        # strides=2, padding='same'))
        block_layers.append(self.Upsampling())
        # block_layers.append(self.Conv(filters=nf, kernel_size=kernel_size,
        # strides=1, padding='same'))
        # block_layers.append(layers.Activation(tf.nn.leaky_relu))
        # block_layers.append(self._pixel_norm())

        block_layers.append(
            self.Conv(filters=nf, kernel_size=kernel_size, strides=1, padding="same")
        )
        block_layers.append(layers.Activation(tf.nn.leaky_relu))
        block_layers.append(self._pixel_norm())

        return models.Sequential(block_layers, name=name)

    def generator_head(self, x, y, alpha):

        x = self.Upsampling()(x)
        x = self.HeadConv1(x)

        y = self.HeadConv2(y)

        output = self._weighted_sum()([x, y, alpha])
        output = layers.Activation("tanh")(output)

        return output

    def add_resolution(self):

        self.current_resolution += 1
        self.resolution_blocks.append(self.highest_resolution_block)
        self.highest_resolution_block = self._make_generator_block(
            self._nf(self.current_resolution),
            name="g_block_{}".format(self.current_resolution),
        )

        self.HeadConv1 = self.Conv(filters=self.num_channels, kernel_size=1)
        self.HeadConv2 = self.Conv(filters=self.num_channels, kernel_size=1)

        self.build([(None, self.latent_size), (1,)])

    def call(self, inputs):
        latents, alpha = inputs
        x = self.generator_base(latents)

        for g_block in self.resolution_blocks:
            x = g_block(x)

        y = self.highest_resolution_block(x)
        return self.generator_head(x, y, alpha)

    def generate(self, latents):
        alpha = tf.constant([1.0], tf.float32)
        image = self.call([latents, alpha])
        image = _adjust_dynamic_range(image, [-1, 1], [0, 255])
        return {"generated": image}

    def save(self, filepath, **kwargs):
        # force retrace of tf.function to track new and deleted variables
        super().save(
            filepath,
            signatures=tf.function(
                self.generate,
                input_signature=(
                    tf.TensorSpec(shape=[None, self.latent_size], dtype=tf.float32),
                ),
            ),
            **kwargs
        )


class Discriminator(tf.keras.Model):
    """
    Progressive Discriminator
    """

    def __init__(
        self,
        label_size=0,
        num_channels=1,
        fmap_base=8192,
        fmap_max=512,
        dimensionality=3,
    ):
        super(Discriminator, self).__init__()

        self.label_size = label_size

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.num_channels = num_channels
        self.dimensionality = dimensionality

        self.current_resolution = 2

        self.Conv = getattr(layers, "Conv{}D".format(self.dimensionality))
        self.AveragePooling = getattr(
            layers, "AveragePooling{}D".format(self.dimensionality)
        )

        self.highest_resolution_block = self._make_discriminator_block(
            self._nf(self.current_resolution - 1),
            name="d_block_{}".format(self.current_resolution),
        )
        self.resolution_blocks = []

        self.BaseConv = self.Conv(
            filters=self._nf(self.current_resolution - 1), kernel_size=1, padding="same"
        )
        self.FadeConv = self.Conv(
            filters=self._nf(self.current_resolution), kernel_size=1, padding="same"
        )
        self.HeadDense1 = tf.keras.layers.Dense(units=self._nf(1))
        self.HeadDense2 = tf.keras.layers.Dense(units=1 + self.label_size)

        images_shape = (
            (None,)
            + (int(2.0 ** self.current_resolution),) * self.dimensionality
            + (self.num_channels,)
        )
        alpha_shape = (1,)

        self.build([images_shape, alpha_shape])

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

    def discriminator_base(self, x, y, alpha):

        x = self.AveragePooling()(x)
        x = self.BaseConv(x)

        lerp_input = self._weighted_sum()([x, y, alpha])

        return lerp_input

    def _make_discriminator_block(self, nf, kernel_size=4, name=""):
        """
        Creates a discriminator block
        """

        block_layers = []

        # block_layers.append(self.Conv(filters=nf, kernel_size=kernel_size,
        # strides=1, padding='same'))
        # block_layers.append(layers.Activation(tf.nn.leaky_relu))

        block_layers.append(
            self.Conv(filters=nf, kernel_size=kernel_size, strides=2, padding="same")
        )
        block_layers.append(layers.Activation(tf.nn.leaky_relu))

        return models.Sequential(block_layers, name=name)

    def discriminator_head(self, x):
        """
        Creates the end of the disciminator with inputs of image dimensions and the
        alpha value to determine level of interpolation. The output node gives
        the prediction of fakeness of the input image.
        """
        x = layers.Flatten()(x)
        x = self.HeadDense1(x)
        x = layers.Activation(tf.nn.leaky_relu)(x)

        output = self.HeadDense2(x)

        score_output = layers.Lambda(lambda x: x[..., 0])(output)
        label_output = layers.Lambda(lambda x: x[..., 1:])(output)

        return score_output, label_output

    def add_resolution(self):
        """
        Adds a resolution step to the trainable discriminator by creating a
        discriminator block and inserting at the start of the existing discriminator
        to increase the resolution of the images discimrinated by a factor of 2.
        The new layers are faded in and controlled by the alpha parameter.
        """

        self.current_resolution += 1

        self.resolution_blocks.append(self.highest_resolution_block)
        self.highest_resolution_block = self._make_discriminator_block(
            self._nf(self.current_resolution - 1),
            name="d_block_{}".format(self.current_resolution),
        )

        self.BaseConv = self.Conv(
            filters=self._nf(self.current_resolution - 1), kernel_size=1, padding="same"
        )
        self.FadeConv = self.Conv(
            filters=self._nf(self.current_resolution), kernel_size=1, padding="same"
        )

        images_shape = (
            (None,)
            + (int(2.0 ** self.current_resolution),) * self.dimensionality
            + (self.num_channels,)
        )
        alpha_shape = (1,)

        self.build([images_shape, alpha_shape])

    def call(self, inputs):

        images, alpha = inputs

        # To bring to the right number of filters
        x = self.FadeConv(images)
        y = self.highest_resolution_block(x)
        x = self.discriminator_base(images, y, alpha)

        for d_block in self.resolution_blocks[::-1]:
            x = d_block(x)

        return self.discriminator_head(x)
