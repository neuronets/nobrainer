import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class BernoulliDropout(tfkl.Layer):
    """Bernoulli dropout layer.

    Outputs the input element multiplied by a random variable
    sampled from a Bernoulli distribution with either mean keep_prob
    (scale_during_training False) or mean 1 (scale_during_training True)

    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        scale_during_training : A boolean value determining whether scaling is performed during training or testing
        mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
        name : A name for this layer (optional).
    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
    """
    def __init__(self, rate, is_monte_carlo, scale_during_training=True, name='bernoulli_dropout'):
        self.rate = rate
        self.is_monte_carlo = is_monte_carlo
        self.scale_during_training = scale_during_training
        self.keep_prob = 1.0 - rate
        super().__init__(self, name=name)

    def call(self, x):
        inference = x

        def apply_bernoulli_dropout():
            d = tf.nn.dropout(inference, rate=self.rate)
            return d * self.keep_prob if self.scale_during_training else d

        if self.scale_during_training:
            expectation = inference
        else:
            expectation = self.keep_prob * inference

        if self.is_monte_carlo:
            return apply_bernoulli_dropout()
        else:
            return expectation


class ConcreteDropout(tfkl.Layer):
    """Concrete Dropout.
    Outputs the input element multiplied by a random variable sampled from a concrete distribution
    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        mc : A boolean Tensor correponding to whether or not Monte-Carlo sampling will be used to calculate the networks output
        name : A name for this layer (optional).
    References:
        Concrete Dropout.
       Y. Gal, J. Hron & A. Kendall,
       Advances in Neural Information Processing Systems.
       2017.
    Links:
      [http://papers.nips.cc/paper/6949-concrete-dropout.pdf]
        (http://papers.nips.cc/paper/6949-concrete-dropout.pdf)
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
            return tf.multiply(x, z)

        if self.n_filters is not None:
            expectation = tf.multiply(self.p, inference)
        else:
            expectation = tf.scalar_mul(self.p, inference)
        if not self.use_expectation:
            expectation = inference
        inference = tf.cond(self.is_monte_carlo, apply_concrete_dropout, lambda: expectation)
        return inference
