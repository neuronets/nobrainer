"""Base classes for all estimators."""

import inspect
import os
from pathlib import Path
import pickle as pk

import tensorflow as tf


def get_strategy(multi_gpu):
    if multi_gpu:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


class BaseEstimator:
    """Base class for all high-level models in Nobrainer."""

    state_variables = []
    model_ = None

    def __init__(self, checkpoint_filepath=None, multi_gpu=True):
        self.checkpoint_tracker = None
        if checkpoint_filepath:
            from .checkpoint import CheckpointTracker

            self.checkpoint_tracker = CheckpointTracker(self, checkpoint_filepath)

        self.strategy = get_strategy(multi_gpu)

    @property
    def model(self):
        return self.model_

    def save(self, save_dir):
        """Saves a trained model"""
        if self.model_ is None:
            raise ValueError("Model is undefined. Please train or load a model")
        self.model_.save(save_dir)
        model_info = {"classname": self.__class__.__name__, "__init__": {}}
        for key in inspect.signature(self.__init__).parameters:
            # TODO this assumes that all parameters passed to __init__
            # are stored as members, which doesn't leave room for
            # parameters that are specific to the runtime context.
            # (e.g. multi_gpu).
            if key == "multi_gpu" or key == "checkpoint_filepath":
                continue
            model_info["__init__"][key] = getattr(self, key)
        for val in self.state_variables:
            model_info[val] = getattr(self, val)
        model_file = Path(save_dir) / "model_params.pkl"
        with open(model_file, "wb") as fp:
            pk.dump(model_info, fp)

    @classmethod
    def load(
        cls,
        model_dir,
        multi_gpu=True,
        custom_objects=None,
        compile=False,
    ):
        """Loads a trained model from a save directory"""
        model_dir = Path(str(model_dir).rstrip(os.pathsep))
        assert model_dir.exists() and model_dir.is_dir()
        model_file = model_dir / "model_params.pkl"
        with open(model_file, "rb") as fp:
            model_info = pk.load(fp)
        if model_info["classname"] != cls.__name__:
            raise ValueError(f"Model class does not match {cls.__name__}")
        del model_info["classname"]

        klass = cls(**model_info["__init__"])
        del model_info["__init__"]
        for key, value in model_info.items():
            setattr(klass, key, value)

        klass.strategy = get_strategy(multi_gpu)
        with klass.strategy.scope():
            klass.model_ = tf.keras.models.load_model(
                model_dir,
                custom_objects=custom_objects,
                compile=compile,
            )
        return klass

    @classmethod
    def init_with_checkpoints(
        cls,
        model_name,
        checkpoint_filepath,
        multi_gpu=True,
        custom_objects=None,
        compile=False,
        model_args=None,
    ):
        """Initialize a model for training, either from the latest
        checkpoint found, or from scratch if no checkpoints are
        found. This is useful for long-running model fits that may be
        interrupted or preepmted during training and need to pick up
        where they left off.

        model_name: str or Module in nobrainer.models, the base model
        for this estimator.

        checkpoint_filepath: str, path to which checkpoints will be
        saved and loaded. Supports the epoch and block flormating
        parameters supported by tensorflows ModelCheckpoint,
        e.g. <path_to_checkpoint_dir>/{epoch:03d}

        """
        from .checkpoint import CheckpointTracker

        checkpoint_tracker = CheckpointTracker(cls, checkpoint_filepath)
        estimator = checkpoint_tracker.load(
            multi_gpu=multi_gpu,
            custom_objects=custom_objects,
            compile=compile,
        )
        if not estimator:
            estimator = cls(model_name, model_args=model_args)
        estimator.checkpoint_tracker = checkpoint_tracker
        checkpoint_tracker.estimator = estimator
        return estimator


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.
        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)
