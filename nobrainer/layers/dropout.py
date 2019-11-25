"""Dropout layers in `tf.keras`."""

import math

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

# TODO: add `K.in_train_phase`.
# TODO: add `get_config` and `compute_output_shape` instance methods.


class BernoulliDropout(tfkl.Layer):
    """Bernoulli dropout layer.

    Outputs the input element multiplied by a random variable
    sampled from a Bernoulli distribution with either mean keep_prob
    (scale_during_training False) or mean 1 (scale_during_training True)

    Parameters
    ----------
    rate : float between 0 and 1, drop probability.
    is_monte_carlo : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
    scale_during_training : A boolean value determining whether scaling is performed during training or testing
    name : A name for this layer (optional).

    References
    ----------
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links
    -----
    [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """

    def __init__(self, rate, is_monte_carlo, scale_during_training=True, name='bernoulli_dropout'):
        self.rate = rate
        self.is_monte_carlo = is_monte_carlo
        self.scale_during_training = scale_during_training
        self.keep_prob = 1.0 - rate
        super().__init__(self, name=name)

    def call(self, x):

        def apply_bernoulli_dropout():
            d = tf.nn.dropout(x, rate=self.rate)
            return d if self.scale_during_training else d * self.keep_prob

        if self.is_monte_carlo:
            return apply_bernoulli_dropout()

        return x if self.scale_during_training else self.keep_prob * x


class ConcreteDropout(tfkl.Layer):
    """Concrete Dropout.

    Outputs the input element multiplied by a random variable sampled from a concrete distribution

    Parameters
    ----------
    is_monte_carlo : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
    temperature
    use_expectation
    name : A name for this layer (optional).

    References
    ----------
    Concrete Dropout. Y. Gal, J. Hron & A. Kendall, Advances in Neural Information Processing Systems. 2017.

    Links
    -----
    [http://papers.nips.cc/paper/6949-concrete-dropout.pdf](http://papers.nips.cc/paper/6949-concrete-dropout.pdf)
    """

    def __init__(self, is_monte_carlo, n_filters=None, temperature=0.02, use_expectation=True, name='concrete_dropout'):
        self.is_monte_carlo = is_monte_carlo
        self.n_filters = n_filters
        self.temperature = temperature
        self.use_expectation = use_expectation
        super().__init__(self, name=name)

    def build(self, input_shape):
        initial_p = tfk.initializers.Constant(0.9)
        if self.n_filters is not None:

            self.p = self.add_weight("p", shape=[self.n_filters], initializer=initial_p)
        else:
            self.p = self.add_weight("p", shape=[], initializer=initial_p)
        self.p = tf.clip_by_value(self.p, 0.05, 0.95)

    def call(self, x):
        inference = x
        eps = tfk.backend.epsilon()

        def apply_concrete_dropout():
            noise = tf.random_uniform(tf.shape(inference), minval=0, maxval=1, dtype=self.dtype)
            z = tf.nn.sigmoid(
                (tf.log(p + eps) - tf.log(1.0 - p + eps) + tf.log(noise + eps) - tf.log(1.0 - noise + eps))
                / temperature)
            return x * z

        if self.is_monte_carlo:
            return apply_concrete_dropout()
        else:
            return inference * self.p if self.use_expectation else inference


class GaussianDropout(tfkl.Layer):
    """Gaussian Dropout.

    Parameters
    ----------

    References
    ----------
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
    (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links
    -----
    [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """

    def __init__(self, rate, is_monte_carlo, scale_during_training=True, name='gaussian_dropout'):
        self.rate = rate
        self.is_monte_carlo = is_monte_carlo
        super().__init__(self, name=name)

    def call(self, x):

        if self.scale_during_training:
            stddev = math.sqrt(self.rate / (1.0 - self.rate))
        else:
            stddev = math.sqrt(self.rate * (1.0 - self.rate))

        def apply_gaussian_dropout():
            return x * tf.random_normal(tf.shape(x), mean=1.0, stddev=stddev)

        return apply_concrete_dropout() if self.is_monte_carlo else x
