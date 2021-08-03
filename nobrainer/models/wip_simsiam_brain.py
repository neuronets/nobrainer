"""
Extending the SimSiam network architecture to brain volumes
author: Dhritiman Das
"""

import nibabel
import nilearn
from nilearn import plotting
import numpy as np
import tensorflow as tf

import nobrainer

csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)

# To create two sets of unlabeled datasets with different augmentations
pre_train_paths_1 = filepaths[:2]
pre_train_paths_2 = filepaths[2:4]

# create training and evaluation datasets for downstream tasks
train_paths = filepaths[4:9]
evaluate_paths = filepaths[9:]

import matplotlib.pyplot as plt

invalid = nobrainer.io.verify_features_labels(pre_train_paths_1)
assert not invalid

invalid = nobrainer.io.verify_features_labels(pre_train_paths_2)
assert not invalid

invalid = nobrainer.io.verify_features_labels(train_paths)
assert not invalid

invalid = nobrainer.io.verify_features_labels(evaluate_paths)
assert not invalid


# convert pretrain, train and validation data to tf records

# for pretrain
nobrainer.tfrecord.write(
    features_labels=pre_train_paths_1,
    filename_template="data/data-pre-train-1_shard-{shard:03d}.tfrec",
    examples_per_shard=2,
)

nobrainer.tfrecord.write(
    features_labels=pre_train_paths_2,
    filename_template="data/data-pre-train-2_shard-{shard:03d}.tfrec",
    examples_per_shard=2,
)

# for training
nobrainer.tfrecord.write(
    features_labels=train_paths,
    filename_template="data/data-train_shard-{shard:03d}.tfrec",
    examples_per_shard=3,
)

# for validation
nobrainer.tfrecord.write(
    features_labels=evaluate_paths,
    filename_template="data/data-evaluate_shard-{shard:03d}.tfrec",
    examples_per_shard=1,
)

n_classes = 1
batch_size = 1
volume_shape = (256, 256, 256)
block_shape = (64, 64, 64)
n_epochs = None
shuffle_buffer_size = 10
num_parallel_calls = 2

# ----for downstream tasks---------

# create dataset -- train
dataset_train = nobrainer.dataset.get_dataset(
    file_pattern="data/data-train_shard*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

# create dataset -- evaluate

dataset_evaluate = nobrainer.dataset.get_dataset(
    file_pattern="data/data-evaluate_shard*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

# -------create tfrecords for pretrain datasets with two different views---------------

# pretrain dataset 1: with random rigid augmentations

dataset_pretrain_1 = nobrainer.dataset.get_dataset(
    file_pattern="data/data-pre-train-1_shard*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    augment=True,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

# pretrain dataset 2: with no augmentations

dataset_pretrain_2 = nobrainer.dataset.get_dataset(
    file_pattern="data/data-pre-train-2_shard*.tfrec",
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    augment=False,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

# ----------------------------------------------------------

# view shapes
print("dataset_train ", dataset_train)
print("dataset_evaluate ", dataset_evaluate)
print("dataset_pretrain 1 ", dataset_pretrain_1)
print("dataset_pretrain 2 ", dataset_pretrain_2)


# transform pretrain dataset to an unlabeled dataset having only features (or image volumes)
dataset_pretrain_1 = dataset_pretrain_1.map(lambda x, y: x)
dataset_pretrain_2 = dataset_pretrain_2.map(lambda x, y: x)

print("pretrain 1", dataset_pretrain_1)
print("pretrain 2", dataset_pretrain_2)


# ----TODO-spatial and intensity transforms

import nobrainer.intensity_transforms as it
import nobrainer.spatial_transforms as st

# augment_one = (
#       dataset_pretrain_1.map(it.addGaussianNoise)
#       .batch(batch_size)
#       .prefetch(tf.data.AUTOTUNE)
# )

# augment_two = (
#     dataset_pretrain_2.map(it.contrastAdjust)
#     .batch(batch_size)
#     .prefetch(tf.data.AUTOTUNE)
# )

# augment_example = (
#     dataset_train.map(it.minmaxIntensityScaling)
#     .batch(batch_size)
#     .prefetch(tf.data.AUTOTUNE)
# )

# print(augment_one)
# print(augment_two)
# print(augment_example)
# ------------------------------------------------------

augment_one = dataset_pretrain_1
augment_two = dataset_pretrain_2

print("augment_one: ", augment_one)
print("augment_two: ", augment_two)


weight_decay = 0.0005
projection_dim = 2048
latent_dim = 512

from tensorflow.keras import activations, layers, regularizers


# define the encoder and projector: this is built on the highresnet backbone
def encoder():
    resnet = nobrainer.models.highresnet(
        n_classes=n_classes,
        input_shape=(*block_shape, 1),
    )
    print("resnet shape", resnet)

    input = tf.keras.layers.Input(shape=(*block_shape, 1))
    print("first input after resnet", input)

    resnet_out = resnet(input)

    x = layers.GlobalAveragePooling3D(name="backbone_pool")(resnet_out)

    x = layers.Dense(
        projection_dim, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(
        projection_dim, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    output = layers.BatchNormalization()(x)

    encoder_model = tf.keras.Model(input, output, name="encoder")
    return encoder_model


exm_encoder = encoder()
exm_encoder.summary()  # view encoder details

# define predictor


def predictor():
    model = tf.keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            tf.keras.layers.InputLayer((projection_dim,)),
            tf.keras.layers.Dense(
                latent_dim,
                use_bias=False,
                kernel_regularizer=regularizers.l2(weight_decay),
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(projection_dim),
        ],
        name="predictor",
    )
    return model


predictor_ch = predictor()
predictor_ch.summary()


def compute_loss(p, z):
    print("p as input is ", p.shape)
    print("z as input is ", z.shape)
    z = tf.stop_gradient(z)
    print("z after stop gradient is ", z.shape)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    print("p after normalization is ", p.shape)
    print("z after normalization is ", z.shape)

    # Negative cosine similarity loss
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


class SimSiam(tf.keras.Model):
    def __init__(self, encoder, predictor):
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):

        data_one, data_two = data
        print("data_one check", tf.is_tensor(data_one))
        print("data_two check", tf.is_tensor(data_two))

        print("data", data)
        print("data_one", data_one)
        print("data_two", data_two)

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(data_one), self.encoder(data_two)
            print("z1 check", tf.is_tensor(z1))
            print("z2 check", tf.is_tensor(z2))
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            print("p1 check", tf.is_tensor(p1))
            print("p2 check", tf.is_tensor(p2))

            print("z1 ", z1.shape)
            print("z2 ", z2.shape)
            print("p1 ", p1.shape)
            print("p2 ", p2.shape)

            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# zip both the augmented datasets
augment_data = tf.data.Dataset.zip((augment_one, augment_two))
print(augment_data)


pretrain_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(pre_train_paths_1),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size,
)

pretrain_steps

validation_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(evaluate_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size,
)

validation_steps

lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.03, decay_steps=steps
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
)

print("augment one", augment_one)
print("augment two", augment_two)
print("augment data", augment_data)

# Compile model and start training.
EPOCHS = 1  # should be higher say >100
simsiam = SimSiam(encoder(), predictor())
simsiam.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6))
history = simsiam.fit(
    augment_data,
    epochs=EPOCHS,
    steps_per_epoch=pretrain_steps,
    callbacks=[early_stopping],
)

plt.plot(history.history["loss"])
plt.grid()
plt.title("Negative Cosine Similairty")
plt.show()

"""Evaluating the SSL Method: Learn a linear classifier on the frozen features of the backbone model"""

backbone = tf.keras.Model(
    simsiam.encoder.input, simsiam.encoder.get_layer("backbone_pool").output
)
backbone.summary()

print(dataset_train)
print(dataset_evaluate)

y_train = tf.keras.utils.to_categorical(dataset_evaluate)

backbone.trainable = False
inputs = layers.Input((*block_shape, 1))
x = backbone(inputs, training=False)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(64)(x)
outputs = layers.Dense(64, activation="sigmoid")(x)
linear_model = tf.keras.Model(inputs, outputs, name="linear_model")

linear_model.summary()

linear_model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9),  # momentum=0.9),
)

train_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(train_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size,
)

train_steps

history = linear_model.fit(
    dataset_train,
    validation_data=dataset_evaluate,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=validation_steps,
    callbacks=[early_stopping],
)

_, test_acc = linear_model.evaluate(dataset_evaluate)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
