"""Methods for training models."""

import os

import tensorflow as tf

from nobrainer import metrics
from nobrainer import models


def train(model, dataset, optimizer, loss, steps_per_epoch, model_kwds={}, n_epochs=1, initial_epoch=0, metrics=None, callbacks=None, validation_dataset=None, validation_steps=None, multi_gpu=False, devices=None):
    """Train a model.

    """

    # TODO: add checks.
    n_classes = dataset.output_shapes[1][-1].value  # last dimension of labels
    input_shape = dataset.output_shapes[0][1:]  # remove batch dimension

    if metrics is None:
        metrics = _get_default_metrics(n_classes)

    if multi_gpu:
        model = _compile_multi_gpu_model(
            model=model,
            n_classes=n_classes,
            input_shape=input_shape,
            optimizer=optimizer,
            loss=loss,
            model_kwds=model_kwds,
            metrics=metrics,
            devices=devices)
    else:
        if isinstance(model, str):
            model_fn = models.get(model)
            model = model_fn(n_classes=n_classes, input_shape=input_shape, **model_kwds)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        elif isinstance(mode, tf.keras.Model):
            # TODO: can we test if the model is compiled? We lose the optimizer
            # state if it was already compiled.
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = _train_generic(
        compiled_model=model,
        dataset=dataset,
        steps_per_epoch=steps_per_epoch,
        n_epochs=n_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        validation_dataset=validation_dataset,
        validation_steps=validation_steps)

    return history


def _train_generic(compiled_model, dataset, steps_per_epoch, n_epochs, initial_epoch, callbacks=None, validation_dataset=None, validation_steps=None):
    """"""

    modelcheckpoint_path = './checkpoints/model-{}_epoch-{{epoch:03d}}.ckpt'.format(compiled_model.name)
    if callbacks is None:
        tensorboard_dir = './logs'
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(os.path.dirname(modelcheckpoint_path), exist_ok=True)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                modelcheckpoint_path, save_weights_only=True),
            # Write metrics out every 100 samples. Writing too frequently can
            # slow down training.
            tf.keras.callbacks.TensorBoard(
                tensorboard_dir, write_graph=True, update_freq=100),
        ]

    # TODO: if we can load weights after compiling model, load the most recent
    # checkpoint before training.
    try:
        compiled_model.load_weights(modelcheckpoint_path)
    except tf.errors.NotFoundError:
        pass

    history = compiled_model.fit(
        dataset,
        epochs=n_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks)

    return history


def _compile_multi_gpu_model(model, n_classes, input_shape, optimizer, loss, model_kwds={}, metrics=None, devices=None):
    """Train a model across multiple GPUs on the same machine."""

    if isinstance(model, tf.keras.Model):
        raise ValueError(
            "Model cannot be a `tf.keras.Model` object, because it must be"
            " instantiated within the MirroredStrategy scope.")

    model_fn = models.get(model)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = model_fn(n_classes=n_classes, input_shape=input_shape, **model_kwds)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def _train_tpu():
    """"""
    raise NotImplementedError()


def _get_default_metrics(n_classes):
    if n_classes == 1:
        m = [
            metrics.dice,
        ]
    elif n_classes > 2:
        m = [
            metrics.generalized_dice,
            metrics.tversky,
        ]
    else:
        raise ValueError("Invalid number of classes. Got {}".format(n_classes))

    return m
