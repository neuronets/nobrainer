"""Model definition for ProgressiveGAN.
"""
import math

import tensorflow as tf
from tensorflow.keras import layers, models


def progressivegan(
    latent_size,
    label_size=0,
    num_channels=1,
    dimensionality=3,
    g_fmap_base=8192,
    d_fmap_base=8192,
    g_fmap_max=256,
    d_fmap_max=256,
    name="progressivegan",
):
    """Instantiate ProgressiveGAN Architecture.

    Parameters
    ----------
    latent_size: 

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


class Generator:
    '''
    Progressive Generator
    '''
    def __init__(self, latent_size, label_size=0, num_channels=1, fmap_base=8192, fmap_max=256, dimensionality=3):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.label_size = label_size

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.num_channels = num_channels
        self.dimensionality = dimensionality

        self.growing_generator = self._make_generator_base()
        self.train_generator = self.growing_generator

        self.current_resolution = 1

        self.Conv =  getattr(layers, 'Conv{}D'.format(self.dimensionality))
        self.ConvTranspose = getattr(layers, 'Conv{}DTranspose'.format(self.dimensionality))
        self.Upsampling = getattr(layers, 'UpSampling{}D'.format(self.dimensionality))

    def _pixel_norm(self, epsilon=1e-8):
        return layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon))
        # return layers.BatchNormalization(axis=-1)

    def _weighted_sum(self):
        return layers.Lambda(lambda inputs : (1-inputs[2])*inputs[0] + (inputs[2])*inputs[1])

    def _nf(self, stage): 
        return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

    def _make_generator_base(self):

        latents = layers.Input(shape=[self.latent_size+self.label_size], name='latents')
        alpha = layers.Input(shape=[], dtype=tf.float32, name='g_alpha')

        # Latents stage
        x = self._pixel_norm()(latents)
        x = layers.Dense(self._nf(1)*(2**self.dimensionality))(x)
        x = layers.Reshape([2]*self.dimensionality+[self._nf(1)])(x)

        return models.Model(inputs=[latents, alpha], outputs=[x], name='generator_base')

    def _make_generator_block(self, nf, name=''):
        
        block_layers = []

        # block_layers.append(self.ConvTranspose(nf, kernel_size=3, strides=2, padding='same'))
        block_layers.append(self.Upsampling())
        block_layers.append(self.Conv(nf, kernel_size=4, strides=1, padding='same'))
        block_layers.append(layers.Activation(tf.nn.leaky_relu))
        block_layers.append(self._pixel_norm())

        # block_layers.append(self.Conv(nf, kernel_size=3, strides=1, padding='same'))
        block_layers.append(self.Conv(nf, kernel_size=4, strides=1, padding='same'))
        block_layers.append(layers.Activation(tf.nn.leaky_relu))
        block_layers.append(self._pixel_norm())

        return models.Sequential(block_layers, name=name)

    def add_resolution(self):

        self.current_resolution += 1

        # Residual from input
        to_rgb_1 = self.Upsampling()(self.growing_generator.output)
        to_rgb_1 = self.Conv(self.num_channels, kernel_size=1)(to_rgb_1)
       
        # Growing generator
        g_block = self._make_generator_block(self._nf(self.current_resolution), name='g_block_{}'.format(self.current_resolution))
        g_block_output = g_block(self.growing_generator.output)
        to_rgb_2 = self.Conv(self.num_channels, kernel_size=1)(g_block_output)

        lerp_output = self._weighted_sum()([to_rgb_1, to_rgb_2, self.growing_generator.input[1]])
        output = layers.Activation('tanh')(lerp_output)

        self.growing_generator = models.Model(inputs=self.growing_generator.input, outputs=g_block_output)
        self.train_generator = models.Model(inputs=self.growing_generator.input, outputs=[output])

    def get_current_resolution(self):
        return self.current_resolution

    def get_trainable_generator(self):
        return self.train_generator

    def get_inference_generator(self):
        raise NotImplementedError


class Discriminator:
    '''
    Progressive Discriminator
    '''

    def __init__(self, label_size=0, num_channels=1, fmap_base=8192, fmap_max=512, dimensionality=3):
        super(Discriminator, self).__init__()

        self.label_size = label_size

        self.fmap_base = fmap_base
        self.fmap_max = fmap_max
        self.num_channels = num_channels
        self.dimensionality = dimensionality

        self.growing_discriminator = self._make_discriminator_base()
        self.train_discriminator = self.growing_discriminator

        self.current_resolution = 1

        self.Conv =  getattr(layers, 'Conv{}D'.format(self.dimensionality))
        self.AveragePooling =  getattr(layers, 'AveragePooling{}D'.format(self.dimensionality))

    def _weighted_sum(self):
        return layers.Lambda(lambda inputs : (1-inputs[2])*inputs[0] + (inputs[2])*inputs[1])

    def _nf(self, stage): 
        return min(int(self.fmap_base / (2.0 ** (stage))), self.fmap_max)

    def _make_discriminator_base(self):

        inputs = layers.Input(shape=(2,)*self.dimensionality + (self._nf(1),), name='dummy')

        x = layers.Flatten()(inputs)
        x = layers.Dense(self._nf(1))(x)
        x = layers.Activation(tf.nn.leaky_relu)(x)

        output = layers.Dense(1+self.label_size)(x)

        return models.Model(inputs=[inputs], outputs=[output])

    def _make_discriminator_block(self, nf, name=''):

        block_layers = []

        block_layers.append(self.Conv(nf, kernel_size=3, strides=1, padding='same'))
        block_layers.append(layers.Activation(tf.nn.leaky_relu))

        block_layers.append(self.Conv(nf, kernel_size=4, strides=2, padding='same'))
        block_layers.append(layers.Activation(tf.nn.leaky_relu))

        return models.Sequential(block_layers, name=name)

    def add_resolution(self):

        self.current_resolution += 1

        inputs = layers.Input(shape=(int(2.0**self.current_resolution),)*self.dimensionality + (self.num_channels,), name='image')
        alpha = layers.Input(shape=[], name='d_alpha')

        # Residual from input
        from_rgb_1 = self.AveragePooling()(inputs)
        from_rgb_1 = self.Conv(self._nf(self.current_resolution-1), kernel_size=1, padding='same', name='from_rgb_1')(from_rgb_1)

        # Growing discriminator
        d_block = self._make_discriminator_block(self._nf(self.current_resolution-1), name='d_block_{}'.format(self.current_resolution))
        from_rgb_2 = self.Conv(self._nf(self.current_resolution), kernel_size=1, padding='same', name='from_rgb_2')(inputs)
        from_rgb_2 = d_block(from_rgb_2)

        lerp_input = self._weighted_sum()([from_rgb_1, from_rgb_2, alpha])

        output = self.growing_discriminator(lerp_input)

        score_output = layers.Lambda(lambda x: x[...,0])(output)
        label_output = layers.Lambda(lambda x: x[...,1:])(output)

        self.growing_discriminator = tf.keras.Sequential([d_block, self.growing_discriminator])
        self.train_discriminator = models.Model(inputs=[inputs, alpha], outputs=[score_output, label_output])

    def get_current_resolution(self):
        return self.current_resolution

    def get_trainable_discriminator(self):
        return self.train_discriminator

    def get_inference_discriminator(self):
        raise NotImplementedError