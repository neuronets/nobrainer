import importlib
import os

import tensorflow as tf

from .base import BaseEstimator
from .. import losses, metrics
from ..dataset import get_steps_per_epoch


class Segmentation(BaseEstimator):
    """Perform segmentation type operations"""

    state_variables = ["block_shape_", "volume_shape_", "scalar_labels_"]

    def __init__(self, base_model, model_args=None, multi_gpu=False):
        super().__init__(multi_gpu=multi_gpu)

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
        warm_start=False,
        # TODO: figure out whether optimizer args should be flattened
        optimizer=None,
        opt_args=None,
        loss=losses.dice,
        metrics=metrics.dice,
    ):
        """Train a segmentation model"""
        # TODO: check validity of datasets

        batch_size = dataset_train.element_spec[0].shape[0]
        self.block_shape_ = dataset_train.block_shape
        self.volume_shape_ = dataset_train.volume_shape
        self.scalar_labels_ = dataset_train.scalar_labels
        n_classes = dataset_train.n_classes
        opt_args = opt_args or {}
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam
            opt_args_tmp = dict(learning_rate=1e-04)
            opt_args_tmp.update(**opt_args)
            opt_args = opt_args_tmp

        def _create(base_model):
            # Instantiate and compile the model
            self.model_ = base_model(
                n_classes=n_classes,
                input_shape=(*self.block_shape_, 1),
                **self.model_args
            )

        def _compile():
            self.model_.compile(
                optimizer(**opt_args),
                loss=loss,
                metrics=metrics,
            )

        if warm_start:
            if self.model is None:
                raise ValueError("warm_start requested, but model is undefined")
            with self.strategy.scope():
                _compile()
        else:
            mod = importlib.import_module("..models", "nobrainer.processing")
            base_model = getattr(mod, self.base_model)
            if batch_size % self.strategy.num_replicas_in_sync:
                raise ValueError("batch size must be a multiple of the number of GPUs")

            with self.strategy.scope():
                _create(base_model)
                _compile()
        print(self.model_.summary())

        # TODO add checkpoint
        self.model_.fit(
            dataset_train,
            epochs=epochs,
            steps_per_epoch=dataset_train.get_steps_per_epoch(batch_size),
            validation_data=dataset_validate,
            validation_steps=dataset_validate.get_steps_per_epoch(batch_size),
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
