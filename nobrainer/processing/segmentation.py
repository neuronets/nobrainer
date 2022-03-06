import importlib
import os

import tensorflow as tf

from .base import BaseEstimator
from .. import losses, metrics
from ..dataset import get_steps_per_epoch


class Segmentation(BaseEstimator):
    """Perform segmentation type operations"""

    state_variables = ["block_shape_", "volume_shape_", "scalar_labels_"]

    def __init__(self, base_model, model_args=None):
        if not isinstance(base_model, str):
            self.base_model = base_model.__name__
        else:
            self.base_model = base_model
        self.model_ = None
        self.model_args = model_args or {}
        self.block_shape_ = None
        self.volume_shape_ = None
        self.scalar_labels_ = None

    def fit(
        self,
        dataset_train,
        dataset_validate=None,
        epochs=1,
        checkpoint_dir=os.getcwd(),
        multi_gpu=False,
        warm_start=False,
        # TODO: figure out whether optimizer args should be flattened
        optimizer=None,
        opt_args=None,
        loss=losses.dice,
        metrics=metrics.dice,
    ):
        """Train a segmentation model"""
        # TODO: check validity of datasets

        # extract dataset information
        batch_size = dataset_train.element_spec[0].shape[0]
        self.block_shape_ = tuple(dataset_train.element_spec[0].shape[1:4])
        self.volume_shape_ = dataset_train.volume_shape
        self.scalar_labels_ = True
        n_classes = 1
        if len(dataset_train.element_spec[1].shape) > 1:
            n_classes = dataset_train.element_spec[1].shape[4]
            self.scalar_labels_ = False
        opt_args = opt_args or {}
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam
            opt_args_tmp = dict(learning_rate=1e-04)
            opt_args_tmp.update(**opt_args)
            opt_args = opt_args_tmp

        def _compile(base_model):
            # Instantiate and compile the model
            model = base_model(
                n_classes=n_classes,
                input_shape=(*self.block_shape_, 1),
                **self.model_args
            )
            model.compile(
                optimizer(**opt_args),
                loss=loss,
                metrics=metrics,
            )
            return model

        if warm_start:
            if self.model is None:
                raise ValueError("warm_start requested, but model is undefined")
        else:
            mod = importlib.import_module("..models", "nobrainer.processing")
            base_model = getattr(mod, self.base_model)
            if multi_gpu:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    self.model_ = _compile(base_model)
            else:
                self.model_ = _compile(base_model)
        print(self.model_.summary())

        train_steps = get_steps_per_epoch(
            n_volumes=dataset_train.n_volumes,
            volume_shape=self.volume_shape_,
            block_shape=self.block_shape_,
            batch_size=batch_size,
        )

        evaluate_steps = None
        if dataset_validate is not None:
            evaluate_steps = get_steps_per_epoch(
                n_volumes=dataset_validate.n_volumes,
                volume_shape=self.volume_shape_,
                block_shape=self.block_shape_,
                batch_size=batch_size,
            )

        # TODO add checkpoint
        self.model_.fit(
            dataset_train,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=dataset_validate,
            validation_steps=evaluate_steps,
        )

        return self

    def predict(self, x, batch_size=1, normalizer=None):
        """Makes a prediction using the trained model"""
        if self.model_ is None:
            raise ValueError("Model is undefined. Please train or load a model")
        from ..prediction import predict

        return predict(
            x,
            self.model_,
            block_shape=self.block_shape_,
            batch_size=batch_size,
            normalizer=normalizer,
        )
