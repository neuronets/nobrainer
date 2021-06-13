"""Dropout layers in `tf.keras`."""

import math

import tensorflow as tf

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
        is_monte_carlo,
        temperature=0.02,
        use_expectation=True,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.is_monte_carlo = is_monte_carlo
        self.temperature = temperature
        self.use_expectation = use_expectation
        self.seed = seed

    def build(self, input_shape):
        initial_p = tfk.initializers.Constant(0.9)
        self.p = self.add_weight(
            name="p", shape=input_shape[-1:], initializer=initial_p, trainable=True
        )
        # TODO: where should this go? Or should it be removed?
        # self.p = tf.clip_by_value(self.p, 0.05, 0.95)

    def call(self, x):
        inference = x
        eps = tfk.backend.epsilon()
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
                    tf.math.log(self.p + eps)
                    - tf.math.log(1.0 - self.p + eps)
                    + tf.math.log(noise + eps)
                    - tf.math.log(1.0 - noise + eps)
                )
                / self.temperature
            )
            return x * z
        else:
            return inference * self.p if self.use_expectation else inference

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "is_monte_carlo": self.is_monte_carlo,
                "temperature": self.temperature,
                "use_expectation": self.use_expectation,
                "seed": self.seed,
            }
        )
        return config


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
