"""Segmentation estimator — scikit-learn-style API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from .base import BaseEstimator


class Segmentation(BaseEstimator):
    """Train and run brain segmentation with a simple API.

    Example::

        seg = Segmentation("unet").fit(dataset, epochs=5)
        result = seg.predict("brain.nii.gz")
        seg.save("my_model")
    """

    state_variables = [
        "base_model",
        "model_args",
        "block_shape_",
        "volume_shape_",
        "n_classes_",
    ]

    def __init__(
        self,
        base_model: str = "unet",
        model_args: dict | None = None,
        checkpoint_filepath: str | Path | None = None,
        multi_gpu: bool = True,
    ):
        super().__init__(checkpoint_filepath, multi_gpu)
        self.base_model = base_model
        self.model_args = model_args or {}
        self.block_shape_: tuple | None = None
        self.volume_shape_: tuple | None = None
        self.n_classes_: int | None = None
        self._optimizer_class: str = "Adam"
        self._optimizer_args: dict = {}
        self._loss_name: str = "unknown"

    def fit(
        self,
        dataset_train: Any,
        dataset_validate: Any | None = None,
        epochs: int = 1,
        optimizer: type = torch.optim.Adam,
        opt_args: dict | None = None,
        loss: Callable | nn.Module | None = None,
        metrics: Callable | None = None,
        callbacks: list | None = None,
    ) -> "Segmentation":
        """Train the model and return self for chaining."""
        from nobrainer.models import get as get_model
        from nobrainer.training import fit as training_fit

        # Store metadata from dataset
        self.block_shape_ = getattr(dataset_train, "block_shape", None)
        self.volume_shape_ = getattr(dataset_train, "volume_shape", None)
        self.n_classes_ = getattr(dataset_train, "n_classes", 1)

        # Set n_classes in model_args
        model_args = {**self.model_args, "n_classes": self.n_classes_}
        factory = get_model(self.base_model)
        self.model_ = factory(**model_args)

        # Configure optimizer
        opt_args = opt_args or {"lr": 1e-3}
        opt = optimizer(self.model_.parameters(), **opt_args)
        self._optimizer_class = optimizer.__name__
        self._optimizer_args = opt_args

        # Configure loss
        if loss is None:
            loss = nn.CrossEntropyLoss()
            self._loss_name = "CrossEntropyLoss"
        elif callable(loss):
            self._loss_name = getattr(loss, "__name__", type(loss).__name__)
            if not isinstance(loss, nn.Module):
                loss = loss()  # factory function like losses.dice()
        else:
            self._loss_name = type(loss).__name__

        # Train
        gpus = torch.cuda.device_count() if self.multi_gpu else 1
        loader = (
            dataset_train.dataloader
            if hasattr(dataset_train, "dataloader")
            else dataset_train
        )
        self._training_result = training_fit(
            model=self.model_,
            loader=loader,
            criterion=loss,
            optimizer=opt,
            max_epochs=epochs,
            gpus=gpus,
            checkpoint_dir=self.checkpoint_filepath,
            callbacks=callbacks,
        )
        self._dataset = dataset_train
        return self

    def predict(
        self,
        x: str | Path | np.ndarray | nib.Nifti1Image,
        batch_size: int = 4,
        block_shape: tuple | None = None,
        normalizer: Callable | None = None,
        n_samples: int = 0,
    ) -> nib.Nifti1Image | tuple[nib.Nifti1Image, ...]:
        """Predict on a volume.

        If ``n_samples > 0`` and model is Bayesian, returns
        ``(label, variance, entropy)`` tuple.
        """
        from nobrainer.prediction import predict, predict_with_uncertainty

        bs = block_shape or self.block_shape_ or (128, 128, 128)

        if n_samples > 0:
            return predict_with_uncertainty(
                inputs=x,
                model=self.model,
                n_samples=n_samples,
                block_shape=bs,
                batch_size=batch_size,
            )
        return predict(
            inputs=x,
            model=self.model,
            block_shape=bs,
            batch_size=batch_size,
            normalizer=normalizer,
        )

    def evaluate(
        self,
        dataset: Any,
        metrics: Callable | None = None,
    ) -> dict:
        """Evaluate model on a dataset. Returns dict with loss and metrics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(device).eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        n_batches = 0
        loader = dataset.dataloader if hasattr(dataset, "dataloader") else dataset

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                else:
                    images, labels = batch[0].to(device), batch[1].to(device)
                pred = self.model_(images)
                total_loss += criterion(pred, labels).item()
                n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "n_batches": n_batches,
        }

    def _build_model(self) -> nn.Module:
        """Reconstruct model architecture from stored metadata."""
        from nobrainer.models import get as get_model

        model_args = {**self.model_args, "n_classes": self.n_classes_}
        return get_model(self.base_model)(**model_args)

    def _restore_from_provenance(self, prov: dict) -> None:
        """Restore state from croissant.json provenance."""
        self.base_model = prov.get("model_architecture", "unet")
        self.model_args = prov.get("model_args", {})
        self.n_classes_ = prov.get("n_classes", 1)
        self.block_shape_ = tuple(prov.get("block_shape", []))
        self._optimizer_class = prov.get("optimizer", {}).get("class", "Adam")
        self._optimizer_args = prov.get("optimizer", {}).get("args", {})
        self._loss_name = prov.get("loss_function", "unknown")
