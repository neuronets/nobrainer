"""Model definition for Attention U-Net.
Adapted from https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/attention-unet.py
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model


def conv_block(x, num_filters):
    x = L.Conv3D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv3D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    return x


def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = L.MaxPool3D()(x)
    return x, p


def attention_gate(g, s, num_filters):
    Wg = L.Conv3D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv3D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv3D(num_filters, 1, padding="same")(out)
    out = L.Activation("sigmoid")(out)

    return out * s


def decoder_block(x, s, num_filters):
    x = L.UpSampling3D()(x)
    s = attention_gate(x, s, num_filters)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x


def attention_unet(n_classes, input_shape):
    """Inputs"""
    inputs = L.Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)

    b1 = conv_block(p3, 512)

    """ Decoder """
    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    """ Outputs """
    outputs = L.Conv3D(n_classes, 1, padding="same")(d3)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    outputs = layers.Activation(final_activation)(outputs)

    """ Model """
    return Model(inputs=inputs, outputs=outputs, name="Attention_U-Net")


if __name__ == "__main__":
    n_classes = 50
    input_shape = (256, 256, 256, 3)
    model = attention_unet(n_classes, input_shape)
    model.summary()
