"""Base classes for all estimators."""

import inspect
import json
from pathlib import Path

import tensorflow as tf


class BaseEstimator:
    """Base class for all high-level models in Nobrainer."""

    state_variables = []
    model_ = None

    @property
    def model(self):
        return self.model_

    def save(self, file_path):
        """Saves a trained model"""
        if self.model_ is None:
            raise ValueError("Model is undefined. Please train or load a model")
        self.model_.save(file_path)
        # TODO: also save any other parameters
        model_file = Path(file_path) / "model_params.json"
        model_info = {"classname": self.__class__.__name__, "__init__": {}}
        for key in inspect.signature(self.__init__).parameters:
            model_info["__init__"][key] = getattr(self, key)
        for val in self.state_variables:
            model_info[val] = getattr(self, val)
        with open(model_file, "wt") as fp:
            json.dump(model_info, fp)

    @staticmethod
    def load(cls, file_path, multi_gpu=False):
        """Saves a trained model"""
        model_file = Path(file_path) / "model_params.json"
        with open(model_file, "wt") as fp:
            model_info = json.load(fp)
        if model_info["classname"] != cls.__class__.__name__:
            raise ValueError(f"Model class does not match {cls.__class__.__name__}")
        del model_info["classname"]
        klass = cls(**model_info["__init__"])
        del model_info["__init__"]
        for key, value in model_info:
            setattr(klass, key, value)
        if multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.Strategy()
        with strategy.scope():
            klass.model_ = tf.keras.models.load_model(file_path)
        return klass


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
