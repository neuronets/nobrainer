import numpy as np
from skimage.transform import resize
import tensorflow as tf


def Guided_GradCAM_3D(Grad_model, ct_io, Class_index):
    # First outputs target convolution and output
    grad_model = Grad_model
    input_ct_io = tf.expand_dims(ct_io, axis=-1)
    input_ct_io = tf.expand_dims(input_ct_io, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_ct_io)
        loss = predictions[:, Class_index]
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    ##--Guided Gradient Part
    #gate_f = tf.cast(output > 0, "float32")
    #gate_r = tf.cast(grads > 0, "float32")
    guided_grads = (
        tf.cast(output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )

    # Average gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:3], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]

    capi = resize(cam, (ct_io.shape))
    print(capi.shape)
    capi = np.maximum(capi, 0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min())
    return heatmap
