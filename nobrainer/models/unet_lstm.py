import tensorflow as tf
from tf.keras import layers
from tf.keras.regularizers import l2


def unet_lstm(
    n_classes,
    input_shape,
    filters,
    activation="tanh",
    reg_val=1e-08,
    drop_val=0.0,
    drop_val_recur=0.0,
    name="unet_lstm",
):
    """unet_lstm -  A model for the spatial and temporal evolution of 3D fields."""

    batch_norm = False
    concat_axis = -1

    inputs = layers.Input(shape=(input_shape))

    x_layer = layers.ConvLSTM3D(
        filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(inputs)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    conv1 = layers.ConvLSTM2D(
        filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(conv1)

    x_layer = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    x_layer = layers.ConvLSTM2D(
        2 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    conv2 = layers.ConvLSTM2D(
        2 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(conv2)

    x_layer = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    x_layer = layers.ConvLSTM2D(
        4 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    conv3 = layers.ConvLSTM2D(
        4 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(conv3)

    x_layer = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    x_layer = layers.ConvLSTM2D(
        8 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    conv4 = layers.ConvLSTM2D(
        8 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(conv4)

    x_layer = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv4)
    x_layer = layers.ConvLSTM2D(
        16 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    conv5 = layers.ConvLSTM2D(
        16 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(conv5)

    x_layer = layers.UpSampling3D(size=(1, 2, 2))(conv5)
    x_layer = layers.concatenate([x_layer, conv4], axis=concat_axis)

    x_layer = layers.ConvLSTM2D(
        8 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.ConvLSTM2D(
        8 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.UpSampling3D(size=(1, 2, 2))(x_layer)
    x_layer = layers.concatenate([x_layer, conv3], axis=concat_axis)

    x_layer = layers.ConvLSTM2D(
        4 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.ConvLSTM2D(
        4 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.UpSampling3D(size=(1, 2, 2))(x_layer)
    x_layer = layers.concatenate([x_layer, conv2], axis=concat_axis)

    x_layer = layers.ConvLSTM2D(
        2 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.ConvLSTM2D(
        2 * filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.UpSampling3D(size=(1, 2, 2))(x_layer)
    x_layer = layers.concatenate([x_layer, conv1], axis=concat_axis)

    x_layer = layers.ConvLSTM2D(
        filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    x_layer = layers.ConvLSTM2D(
        filters,
        (3, 3),
        activation=activation,
        padding="same",
        kernel_regularizer=l2(reg_val),
        recurrent_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        dropout=drop_val,
        recurrent_dropout=drop_val_recur,
        return_sequences=True,
    )(x_layer)
    if batch_norm:
        x_layer = layers.BatchNormalization(axis=concat_axis)(x_layer)

    outputs = layers.ConvLSTM2D(
        n_classes, (1, 1), activation="linear", padding="same", return_sequences=False
    )(x_layer)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
