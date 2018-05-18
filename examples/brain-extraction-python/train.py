"""Example training a nobrainer model for brain extraction."""

import nobrainer

# Instantiate object to perform real-time data augmentation on training data.
# This object is similar to `keras.preprocessing.image.ImageDataGenerator` but
# works with volumetric data.
volume_data_generator = nobrainer.VolumeDataGenerator(
    samplewise_minmax=True,
    rot90_x=True,
    rot90_y=True,
    rot90_z=True,
    flip_x=True,
    flip_y=True,
    flip_z=True,
    salt_and_pepper=True,
    gaussian=True,
    reduce_contrast=True,
    binarize_y=True)

# Instantiate TensorFlow model.
model = nobrainer.HighRes3DNet(
    n_classes=2,  # Two classes for brain extraction (i.e., brain vs not brain)
    optimizer='Adam',
    learning_rate=0.01,
    # Model-specific options.
    one_batchnorm_per_resblock=True,
    dropout_rate=0.25)

# Read in filepaths to features and labels.
filepaths = nobrainer.read_csv("features_labels.csv")

# Most GPUs do not have enough memory to represent a 256**3 volume during
# training, so we train on blocks of data. Here, we set the shape of the
# blocks.
block_shape = (128, 128, 128)

# Train model.
nobrainer.train(
    model=model,
    volume_data_generator=volume_data_generator,
    filepaths=filepaths,
    volume_shape=(256, 256, 256),
    block_shape=block_shape,
    strides=block_shape,
    batch_size=1,  # number of blocks per training step
    n_epochs=1,  # number of passes through the training set
    prefetch=4)  # prefetch this many full volumes.
