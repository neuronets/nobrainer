import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class BernoulliDropout(tfkl.Layer):
    """ Bernoulli Dropout.
    Outputs the input element multiplied by a random variable sampled from a Bernoulli distribution with either mean keep_prob (scale_during_training False) or mean 1 (scale_during_training True)
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
    def __init__(self, keep_prob, is_monte_carlo, scale_during_training=True, name='bernoulli_dropout'):
        self.keep_prob = keep_prob
        self.is_monte_carlo = is_monte_carlo
        self.scale_during_training = scale_during_training
        self.name = name

    def call(self, x):
        inference = x

        def apply_bernoulli_dropout():
            if self.scale_during_training:
                return tf.nn.dropout(inference, self.keep_prob)
            else:
                return tf.scalar_mul(self.keep_prob, tf.nn.dropout(inference, self.keep_prob))

        if self.scale_during_training:
            expectation =  inference
        else:
            expectation =  tf.scalar_mul(self.keep_prob, inference)
        inference = tf.cond(self.is_monte_carlo, apply_bernoulli_dropout, lambda: expectation)
    return inference


class ConcreteDropout(tfkl.Layer):
    """ Concrete Dropout.
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
        self.name = name

    def build(self, input_shape):
        initial_p = 0.9
        if n_filters is not None:
            self.add_variable("p", shape=[self.n_filters] initializer=initial_p)
        else:
            self.add_variable("p", shape=[] initializer=initial_p)
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
