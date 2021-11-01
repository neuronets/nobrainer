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

        
class BrainSiamese(tf.keras.Model):
    """Self-Supervised Siamese Network Trainer.

    Trains two view - one through an encoder and predictor MLP, and the other through a predictor MLP only. A stop gradient is applied to stabiise the representations.

    Parameters
    ----------
    encoder : tf.keras.Model, based on ResNet; Instantiated using nobrainer.models.brainsiam
    predictor : tf.keras.Model, Dense-MLP; Instantiated using nobrainer.models.brainsiam

    References
    ----------
    Exploring Simple Siamese Representation Learning.
    Xinlei Chen, Kaiming He, CVPR 2021.

    Links
    -----
    [SimSiam-CVPR-2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)
    """
    
    def __init__(self, encoder, predictor):
        super(BrainSiamese, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        data_one, data_two = data #unpacking the data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            proj_1, proj_2 = self.encoder(data_one), self.encoder(data_two)
            pred_1, pred_2 = self.predictor(proj_1), self.predictor(proj_2)
            loss =  compute_loss(pred_1, proj_2) / 2 + compute_loss(pred_2, proj_1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        


def compute_loss(pred, proj):  
    proj = tf.stop_gradient(proj)
    pred = tf.math.l2_normalize(pred, axis=1)
    proj = tf.math.l2_normalize(proj, axis=1)

    # Negative cosine similarity loss
    return -tf.reduce_mean(tf.reduce_sum((pred * proj), axis=1))
