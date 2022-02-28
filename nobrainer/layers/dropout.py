"""Dropout layers in `tf.keras`."""

import math

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers

# TODO: add `K.in_train_phase`.


class BernoulliDropout(tfkl.Layer):
    """Bernoulli dropout layer.

    Outputs the input element multiplied by a random variable
    sampled from a Bernoulli distribution with either mean keep_prob
    (scale_during_training False) or mean 1 (scale_during_training True)

    Parameters
    ----------
    rate : float between 0 and 1, drop probability.
    is_monte_carlo : A boolean Tensor corresponding to whether or not Monte-Carlo
        sampling will be used to calculate the networks output
    scale_during_training : A boolean value determining whether scaling is performed
        during training or testing
    seed : int, value to seed random number generator.

    References
    ----------
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links
    -----
    [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """

    def __init__(
        self, rate, is_monte_carlo, scale_during_training=True, seed=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.rate = rate
        self.is_monte_carlo = is_monte_carlo
        self.scale_during_training = scale_during_training
        self.keep_prob = 1.0 - rate
        self.seed = seed

    def call(self, x):
        if self.is_monte_carlo:
            d = tf.nn.dropout(x, rate=self.rate, seed=self.seed)
            return d if self.scale_during_training else d * self.keep_prob
        return x if self.scale_during_training else self.keep_prob * x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "is_monte_carlo": self.is_monte_carlo,
                "scale_during_training": self.scale_during_training,
                "seed": self.seed,
            }
        )
        return config


def divergence_fn(pl, pr, scale_factor):
    """Divergence computation for concrete dropout"""
    return tf.reduce_sum(
        tf.add(
            tf.multiply(
                pl,
                tf.subtract(
                    tf.math.log(tf.add(pl, tfk.backend.epsilon())),
                    tf.math.log(pr),
                ),
            ),
            tf.multiply(
                tf.subtract(tfk.backend.constant(1), pl),
                tf.subtract(
                    tf.math.log(
                        tf.add(
                            tf.subtract(tfk.backend.constant(1), pl),
                            tfk.backend.epsilon(),
                        )
                    ),
                    tf.math.log(pr),
                ),
            ),
        )
    ) / tf.cast(scale_factor, dtype=tf.float32)


class ConcreteDropout(tfkl.Layer):
    """Concrete Dropout.
    Outputs the input element multiplied by a random variable sampled from a concrete
    distribution
    Parameters
    ----------
    is_monte_carlo : A boolean Tensor corresponding to whether or not Monte-Carlo
        sampling will be used to calculate the networks output temperature.
    use_expectation : boolean
    seed : int, value to seed random number generator.
    name : A name for this layer (optional).
    References
    ----------
    Concrete Dropout. Y. Gal, J. Hron & A. Kendall, Advances in Neural Information
    Processing Systems. 2017.
    Links
    -----
    [http://papers.nips.cc/paper/6949-concrete-dropout.pdf](http://papers.nips.cc/paper/6949-concrete-dropout.pdf)
    """

    def __init__(
        self,
        is_monte_carlo=False,
        temperature=0.02,
        use_expectation=False,
        scale_factor=1,
        seed=None,
        **kwargs
    ):
        super(ConcreteDropout, self).__init__(**kwargs)
        self.is_monte_carlo = is_monte_carlo
        self.temperature = temperature
        self.use_expectation = use_expectation
        self.seed = seed
        self.scale_factor = scale_factor

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        # if self.concrete:
        p_prior = tfk.initializers.Constant(0.5)
        initial_p = tfk.initializers.Constant(0.9)
        self.p_post = self.add_variable(
            name="p_l", shape=input_shape[-1:], initializer=initial_p, trainable=True
        )

        self.p_prior = self.add_variable(
            name="pr", shape=input_shape[-1:], initializer=p_prior, trainable=False
        )
        self.p_post = tfk.backend.clip(self.p_post, 0.05, 0.95)
        self.built = True

    def call(self, x):
        outputs = self._apply_concrete(x)
        self._apply_divergence_concrete(self.scale_factor, name="concrete_loss")
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "is_monte_carlo": self.is_monte_carlo,
                "temperature": self.temperature,
                "use_expectation": self.use_expectation,
                "seed": self.seed,
                "scale": self.scale_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        return cls(**config)

    def _apply_concrete(self, inp):
        inference = inp
        eps = tfk.backend.epsilon()
        use_expectation = self.use_expectation
        if self.is_monte_carlo:
            noise = tf.random.uniform(
                tf.shape(inference),
                minval=0,
                maxval=1,
                seed=self.seed,
                dtype=self.dtype,
            )
            z = tf.nn.sigmoid(
                (
                    tf.math.log(self.p_post + eps)
                    - tf.math.log(1.0 - self.p_post + eps)
                    + tf.math.log(noise + eps)
                    - tf.math.log(1.0 - noise + eps)
                )
                / self.temperature
            )
            return inp * z
        else:
            return inference * self.p_post if use_expectation else inference

    def _apply_divergence_concrete(self, scale_factor, name):
        divergence = tf.identity(
            divergence_fn(self.p_post, self.p_prior, scale_factor), name=name
        )
        self.add_loss(divergence)


class GaussianDropout(tfkl.Layer):
    """Gaussian Dropout.

    Parameters
    ----------
    rate : float between 0 and 1, drop probability.
    is_monte_carlo : A boolean Tensor corresponding to whether or not Monte-Carlo
        sampling will be used to calculate the networks output
    scale_during_training : A boolean value determining whether scaling is performed
        during training or testing
    seed : int, value to seed random number generator.

    References
    ----------
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
    (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links
    -----
    [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """

    def __init__(
        self, rate, is_monte_carlo, scale_during_training=True, seed=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.rate = rate
        self.is_monte_carlo = is_monte_carlo
        self.scale_during_training = scale_during_training
        self.seed = seed

    def call(self, x):
        if self.is_monte_carlo:
            if self.scale_during_training:
                stddev = math.sqrt(self.rate / (1.0 - self.rate))
            else:
                stddev = math.sqrt(self.rate * (1.0 - self.rate))
            noise = tf.random.normal(
                tf.shape(x), mean=1.0, stddev=stddev, seed=self.seed
            )
            return x * noise
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "is_monte_carlo": self.is_monte_carlo,
                "scale_during_training": self.scale_during_training,
                "seed": self.seed,
            }
        )
        return config
