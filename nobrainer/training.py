"""training utilities (supports training on single and multiple GPUs)"""

import os

import tensorflow as tf
from tensorflow.python.keras.engine import compile_utils

from .losses import gradient_penalty
from .volume import adjust_dynamic_range as _adjust_dynamic_range


class ProgressiveGANTrainer(tf.keras.Model):
    """Progressive Generative Adversarial Network Trainer.

    Trains discriminator and generator alternatively in an adversarial manner for generation of
    brain MRI images. Uses a progressive method for each resolution and supports the smooth
    transition of new layers, as explained in the reference.

    Parameters
    ----------
    discriminator : tf.keras.Model, Instantiated using nobrainer.models
    generator : tf.keras.Model, Instantiated using nobrainer.models
    gradient_penalty : boolean, Use gradient penalty on discriminator for smooth training.

    References
    ----------
    Progressive Growing of GANs for Improved Quality, Stability, and Variation.
    T. Karras, T. Aila, S. Laine & J. Lehtinen, International Conference on
    Learning Representations. 2018.

    Links
    -----
    [https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)
    """

    def __init__(self, discriminator, generator, gradient_penalty=False):
        super(ProgressiveGANTrainer, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gradient_penalty = gradient_penalty
        self.latent_size = generator.latent_size
        self.resolution = 8  # Default resolution
        self.train_step_counter = tf.Variable(
            0.0
        )  # For calculating alpha in transition phase
        self.phase = tf.Variable("resolution")  # For determining whether alpha is 1

    def compile(self, d_optimizer, g_optimizer, g_loss_fn, d_loss_fn):
        super(ProgressiveGANTrainer, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.g_loss_fn = compile_utils.LossesContainer(g_loss_fn)
        self.d_loss_fn = compile_utils.LossesContainer(d_loss_fn)

        if self.gradient_penalty:
            self.gradient_penalty_fn = compile_utils.LossesContainer(gradient_penalty)

    def train_step(self, reals):
        if isinstance(reals, tuple):
            reals = reals[0]

        # get batch size dynamically
        batch_size = tf.shape(reals)[0]

        # normalize the real images using minmax to [-1, 1]
        reals = _adjust_dynamic_range(reals, [0.0, 255.0], [-1.0, 1.0])

        # calculate alpha differently for transition and resolution phase
        self.train_step_counter.assign_add(1.0)
        alpha = tf.cond(
            tf.math.equal(self.phase, "transition"),
            lambda: self.train_step_counter / self.steps_per_epoch,
            lambda: tf.constant([1.0]),
        )
        alpha = tf.reshape(alpha, (1,))

        # train discriminator
        latents = tf.random.normal((batch_size, self.latent_size))
        fake_labels = tf.ones((batch_size, 1)) * -1
        real_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fakes = self.generator([latents, alpha])
            fakes_pred, labels_pred_fake = self.discriminator([fakes, alpha])
            reals_pred, labels_pred_real = self.discriminator([reals, alpha])

            fake_loss = self.d_loss_fn(fake_labels, fakes_pred)
            real_loss = self.d_loss_fn(real_labels, reals_pred)
            d_loss = 0.5 * (fake_loss + real_loss)

            # calculate and add the gradient penalty loss using average samples for discriminator
            if self.gradient_penalty:
                weight_shape = (tf.shape(reals)[0],) + (
                    1,
                    1,
                    1,
                    1,
                )  # broadcasting to right shape
                weight = tf.random.uniform(weight_shape, minval=0, maxval=1)
                average_samples = (weight * reals) + ((1 - weight) * fakes)
                average_pred = self.discriminator(([average_samples, alpha]))
                gradients = tf.gradients(average_pred, average_samples)[0]
                gp_loss = self.gradient_penalty_fn(gradients, reals_pred)
                d_loss += gp_loss

        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

        # train generator
        misleading_labels = tf.ones((batch_size, 1))

        latents = tf.random.normal((batch_size, self.latent_size))
        with tf.GradientTape() as tape:
            fakes = self.generator([latents, alpha])
            fakes_pred, labels_pred = self.discriminator([fakes, alpha])

            g_loss = self.g_loss_fn(misleading_labels, fakes_pred)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss}

    def fit(
        self, *args, resolution=8, phase="resolution", steps_per_epoch=300, **kwargs
    ):
        self.resolution = resolution
        self.steps_per_epoch = steps_per_epoch
        self.train_step_counter.assign(0.0)
        self.phase.assign(phase)
        super().fit(*args, steps_per_epoch=steps_per_epoch, **kwargs)

    def save_weights(self, filepath, **kwargs):
        """
        Override base class function to save the weights of the constituent models
        """
        self.generator.save_weights(
            os.path.join(filepath, "g_weights_res_{}.h5".format(self.resolution)),
            **kwargs
        )
        self.discriminator.save_weights(
            os.path.join(filepath, "d_weights_res_{}.h5".format(self.resolution)),
            **kwargs
        )
        self.generator.save(
            os.path.join(filepath, "generator_res_{}".format(self.resolution))
        )


class ProgressiveAETrainer(tf.keras.Model):
    """Progressive Autoencoder Trainer.

    Trains encoder and decoder using reconstruction error to learn latent representations of
    brain MRI images. Uses a progressive method for each resolution and supports the smooth
    transition of new layers, as explained in the reference.

    Parameters
    ----------
    encoder : tf.keras.Model, Instantiated using nobrainer.models
    decoder : tf.keras.Model, Instantiated using nobrainer.models
    fixed : boolean, if True, decoder is fixed and go not undergo training

    References
    ----------
    Progressive Growing of GANs for Improved Quality, Stability, and Variation.
    T. Karras, T. Aila, S. Laine & J. Lehtinen, International Conference on
    Learning Representations. 2018.

    Links
    -----
    [https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)
    """

    def __init__(self, encoder, decoder, fixed=False):
        super(ProgressiveAETrainer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fixed = fixed
        if self.fixed:
            self.decoder.trainable = False

        self.latent_size = encoder.latent_size
        self.resolution = 8
        self.train_step_counter = tf.Variable(0.0)
        self.phase = tf.Variable("resolution")

    def compile(self, optimizer, loss_fn):
        super(ProgressiveAETrainer, self).compile()

        self.optimizer = optimizer
        self.loss_fn = compile_utils.LossesContainer(loss_fn)

    def train_step(self, images):

        if isinstance(images, tuple):
            images = images[0]
        else:

            images = _adjust_dynamic_range(images, [0.0, 255.0], [-1.0, 1.0])

            self.train_step_counter.assign_add(1.0)

            alpha = tf.cond(
                tf.math.equal(self.phase, "transition"),
                lambda: self.train_step_counter / self.steps_per_epoch,
                lambda: tf.constant([1.0]),
            )

            if self.fixed:
                alpha = tf.reshape(alpha, (1,))
                beta = tf.reshape(tf.constant(1.0), (1,))
            else:
                alpha = tf.reshape(alpha, (1,))
                beta = tf.reshape(alpha, (1,))

        with tf.GradientTape() as (tape):
            latent = self.encoder([images, alpha])
            reconstructed = self.decoder([latent, beta])
            loss = tf.math.reduce_mean((self.loss_fn(images, reconstructed)), axis=None)

        gradients = tape.gradient(
            loss, self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_variables + self.decoder.trainable_variables,
            )
        )

        return {"loss": loss}

    def fit(
        self, *args, resolution=8, phase="resolution", steps_per_epoch=300, **kwargs
    ):
        self.resolution = resolution
        self.steps_per_epoch = steps_per_epoch
        self.train_step_counter.assign(0.0)
        self.phase.assign(phase)
        (super().fit)(args, steps_per_epoch=steps_per_epoch, **kwargs)
