"""
Extending the SimSiam network architecture to brain volumes
author: Dhritiman Das
"""

import tensorflow as tf

from tensorflow.keras import layers, regularizers, activations
import nobrainer
import numpy as np

def simsiam_brain(
    n_classes,
    input_shape,
    weight_decay = 0.0005,
    projection_dim = 2048,
    latent_dim = 512,
    name="simsiam_brain",
    **kwargs
):

    print("projection dimension is: ", projection_dim)
    print("latent dimension: ", latent_dim)

    print("may not show both encoder and predictor architecture summary due to lack of space....")

    def encoder():
        resnet = nobrainer.models.highresnet(n_classes=n_classes, input_shape=input_shape)
           
        input = tf.keras.layers.Input(shape=input_shape)

        resnet_out = resnet(input)
        
        x = layers.GlobalAveragePooling3D(name="backbone_pool")(resnet_out)

        x = layers.Dense(
                projection_dim, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay)
            )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dense(
                projection_dim, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay)
            )(x)
        output = layers.BatchNormalization()(x)

        encoder_model = tf.keras.Model(input, output, name="encoder")
        return encoder_model


    def predictor():
        predictor_model = tf.keras.Sequential(
                [
                    # Note the AutoEncoder-like structure.
                    tf.keras.layers.InputLayer((projection_dim,)),
                    tf.keras.layers.Dense(
                        latent_dim, 
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(weight_decay)
                        ),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(projection_dim),
                ],
                name="predictor",
            )
        
        return predictor_model


    encoder_model = encoder()
    predictor_model = predictor()
    
    encoder_model.summary()
    predictor_model.summary()

    return encoder_model, predictor_model
