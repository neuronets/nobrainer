import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def unet_lstm(
    n_classes=1,
    input_shape=(32, 32, 32, 8, 1),
    filters=8,
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
        3,
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

    conv1 = layers.ConvLSTM3D(
        filters,
        3,
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

    # x_layer = layers.MaxPooling4D(pool_size=(1, 2, 2 , 2))(conv1) ToDo
    x_layer = layers.ConvLSTM3D(
        2 * filters,
        3,
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

    conv2 = layers.ConvLSTM3D(
        2 * filters,
        3,
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

    # x_layer = layers.MaxPooling4D(pool_size=(1, 2, 2, 2))(conv2) ToDo
    x_layer = layers.ConvLSTM3D(
        4 * filters,
        3,
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

    conv3 = layers.ConvLSTM3D(
        4 * filters,
        3,
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

    # x_layer = layers.MaxPooling4D(pool_size=(1, 2, 2, 2))(conv3)
    x_layer = layers.ConvLSTM3D(
        8 * filters,
        3,
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

    conv4 = layers.ConvLSTM3D(
        8 * filters,
        3,
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

    # x_layer = layers.MaxPooling4D(pool_size=(1, 2, 2 , 2))(conv4) ToDo
    x_layer = layers.ConvLSTM3D(
        16 * filters,
        3,
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

    conv5 = layers.ConvLSTM3D(
        16 * filters,
        3,
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

    # x_layer = layers.UpSampling4D(size=(1, 2, 2, 2))(conv5) ToDo
    x_layer = layers.concatenate([x_layer, conv4], axis=concat_axis)

    x_layer = layers.ConvLSTM3D(
        8 * filters,
        3,
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

    x_layer = layers.ConvLSTM3D(
        8 * filters,
        3,
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

    # x_layer = layers.UpSampling4D(size=(1, 2, 2, 2))(x_layer) ToDo
    x_layer = layers.concatenate([x_layer, conv3], axis=concat_axis)

    x_layer = layers.ConvLSTM3D(
        4 * filters,
        3,
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

    x_layer = layers.ConvLSTM3D(
        4 * filters,
        3,
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

    # x_layer = layers.UpSampling4D(size=(1, 2, 2, 2))(x_layer) ToDo
    x_layer = layers.concatenate([x_layer, conv2], axis=concat_axis)

    x_layer = layers.ConvLSTM3D(
        2 * filters,
        3,
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

    x_layer = layers.ConvLSTM3D(
        2 * filters,
        3,
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

    # x_layer = layers.UpSampling4D(size=(1, 2, 2))(x_layer) ToDo
    x_layer = layers.concatenate([x_layer, conv1], axis=concat_axis)

    x_layer = layers.ConvLSTM3D(
        filters,
        3,
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

    x_layer = layers.ConvLSTM3D(
        filters,
        3,
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

    outputs = layers.ConvLSTM3D(
        n_classes, 1, activation="linear", padding="same", return_sequences=False
    )(x_layer)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
