'''training utilities (supports training on single and multiple GPUs)'''

from pathlib import Path
from functools import partial
import time

import numpy as np
from PIL import Image
import tensorflow as tf
import nibabel as nib

import os

from tensorflow.python.keras.engine import compile_utils
from nobrainer.volume import adjust_dynamic_range as _adjust_dynamic_range
from nobrainer.losses import gradient_penalty


class ProgressiveGANTrainer(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_size, gradient_penalty=False):
        super(ProgressiveGANTrainer, self).__init__()
        self.discriminator = discriminator 
        self.generator = generator
        self.latent_size = latent_size
        self.train_step_counter = 0 # For calculating alpha in transition phase
        self.phase = 'transition'
        self.gradient_penalty = gradient_penalty

    def compile(self, d_optimizer, g_optimizer, g_loss_fn, d_loss_fn):
        super(ProgressiveGANTrainer, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.g_loss_fn = compile_utils.LossesContainer(g_loss_fn)
        self.d_loss_fn = compile_utils.LossesContainer(d_loss_fn)

        if self.gradient_penalty:
            self.gradient_penalty_fn = compile_utils.LossesContainer(gradient_penalty)

    def train_step(self, reals, phase='transition'):
        if isinstance(reals, tuple):
            reals = reals[0]
        batch_size = tf.shape(reals)[0]

        # reals = _adjust_dynamic_range(reals, [0.0, 255.0], [-1.0, 1.0])

        if phase == 'transition':
            alpha = self.train_step_counter / self.steps_per_epoch
            self.train_step_counter += 1
        else:
            alpha = 1.0

        alpha = tf.constant([alpha], tf.float32)

        latents = tf.random.normal((batch_size, self.latent_size))
        fake_labels = tf.ones((batch_size, 1))*-1
        real_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fakes = self.generator([latents, alpha])
            fakes_pred, labels_pred_fake = self.discriminator([fakes, alpha])
            reals_pred, labels_pred_real = self.discriminator([reals, alpha])

            fake_loss = self.d_loss_fn(fake_labels, fakes_pred)
            real_loss = self.d_loss_fn(real_labels, reals_pred)
            d_loss = 0.5 * (fake_loss + real_loss)

            if self.gradient_penalty:
                weight_shape = (tf.shape(reals)[0],) + (1,1,1,1)
                weight = tf.random.uniform(weight_shape, minval=0, maxval=1)
                average_samples = (weight * reals) + ((1 - weight) * fakes)
                average_pred = self.discriminator(([average_samples, alpha]))
                gradients = tf.gradients(average_pred, average_samples)[0]
                gp_loss = self.gradient_penalty_fn(gradients, reals_pred)
                d_loss += gp_loss

        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        misleading_labels = tf.ones((batch_size, 1))

        latents = tf.random.normal((batch_size, self.latent_size))
        with tf.GradientTape() as tape:
            fakes = self.generator([latents, alpha])
            fakes_pred, labels_pred = self.discriminator([fakes, alpha])

            g_loss = self.g_loss_fn(misleading_labels, fakes_pred)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def fit(self, *args, phase='resolution', resolution=8, steps_per_epoch=300, **kwargs):
        self.phase = phase
        self.train_step_counter = 0
        self.resolution = resolution
        self.steps_per_epoch = steps_per_epoch
        super().fit(*args, steps_per_epoch=steps_per_epoch, **kwargs)

    def save_weights(filepath, **kwargs):
    	self.generator.save_weights(os.path.join(filepath, 'g_{}'.format(self.resolution)), **kwargs)
    	self.discriminator.save_weights(os.path.join(filepath, 'd_{}'.format(self.resolution)), **kwargs)

    def generate_images(self):
    	pass




