"""
Extending the SimSiam network architecture to brain volumes
author: Dhritiman Das
"""

import tensorflow as tf


class BrainSiamese(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(BrainSiamese, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        data_one, data_two = data  # unpacking the data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            proj_1, proj_2 = self.encoder(data_one), self.encoder(data_two)
            pred_1, pred_2 = self.predictor(proj_1), self.predictor(proj_2)
            loss = (
                negative_cosine_loss(pred_1, proj_2) / 2
                + negative_cosine_loss(pred_2, proj_1) / 2
            )

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def negative_cosine_loss(pred, proj):
    proj = tf.stop_gradient(proj)
    pred = tf.math.l2_normalize(pred, axis=1)
    proj = tf.math.l2_normalize(proj, axis=1)

    # Negative cosine similarity loss
    return -tf.reduce_mean(tf.reduce_sum((pred * proj), axis=1))
