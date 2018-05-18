"""Example script to save a TensorFlow model as a SavedModel."""

import nobrainer
import tensorflow as tf

# Set shape of the input block (batch_size, *block_shape, 1). The 1 indicates
# one channel (grayscale).
volume = tf.placeholder(tf.float32, shape=[None, 128, 128, 128, 1])
# Attach the volume placeholder above to the input function for the model.
serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
    'volume': volume})

model = nobrainer.HighRes3DNet(
    n_classes=2,  # Two classes for brain extraction (i.e., brain vs not brain)
    # Model-specific options.
    one_batchnorm_per_resblock=True,
    dropout_rate=0.25)

# Save the model to a .pb file in the specified directory.
savedir = model.export_savedmodel("savedmodel", serving_input_fn)

print("Saved model to {}".format(savedir.decode()))
