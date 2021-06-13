import numpy as np
import tensorflow as tf

from ..padding import ZeroPadding3DChannels


def test_zeropadding3dchannels():
    # This test function is a much shorter version of
    # `tensorflow.python.keras.testing_utils.layer_test`.
    input_data_shape = (4, 32, 32, 32, 1)
    input_data = 10 * np.random.random(input_data_shape)

    x = tf.keras.layers.Input(shape=input_data_shape[1:], dtype=input_data.dtype)
    y = ZeroPadding3DChannels(4)(x)
    model = tf.keras.Model(x, y)

    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert actual_output_shape == (4, 32, 32, 32, 9)
    assert not actual_output[..., :4].any()
    assert actual_output[..., 4].any()
    assert not actual_output[..., 5:].any()

    return actual_output
