"""Generation estimator — scikit-learn-style API for GANs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from .base import BaseEstimator


class Generation(BaseEstimator):
    """Train and generate synthetic brain volumes.

    Example::

        gen = Generation("progressivegan").fit(dataset, epochs=100)
        images = gen.generate(n_images=5)
    """

    state_variables = ["base_model", "model_args", "latent_size"]

    def __init__(
        self,
        base_model: str = "progressivegan",
        model_args: dict | None = None,
        multi_gpu: bool = True,
    ):
        super().__init__(multi_gpu=multi_gpu)
        self.base_model = base_model
        self.model_args = model_args or {}
        self.latent_size = self.model_args.get("latent_size", 256)

    def fit(
        self,
        dataset_train: Any,
        epochs: int = 100,
        **trainer_kwargs: Any,
    ) -> "Generation":
        """Train the generative model using Lightning."""
        import pytorch_lightning as pl

        from nobrainer.models import get as get_model

        factory = get_model(self.base_model)
        self.model_ = factory(**self.model_args)
        self.latent_size = getattr(self.model_, "latent_size", self.latent_size)

        loader = (
            dataset_train.dataloader
            if hasattr(dataset_train, "dataloader")
            else dataset_train
        )

        trainer_defaults = {
            "max_steps": epochs,
            "accelerator": "auto",
            "devices": 1,
            "enable_checkpointing": False,
            "logger": False,
        }
        trainer_defaults.update(trainer_kwargs)

        trainer = pl.Trainer(**trainer_defaults)
        trainer.fit(self.model_, loader)

        self._dataset = dataset_train
        self._training_result = {
            "history": [{"epoch": e, "loss": None} for e in range(1, epochs + 1)],
            "checkpoint_path": None,
        }
        return self

    def generate(
        self,
        n_images: int = 1,
        data_type: type | None = None,
    ) -> list[nib.Nifti1Image]:
        """Generate synthetic brain volumes."""
        self.model_.eval()
        gen = self.model_.generator
        gen.current_level = getattr(gen, "current_level", 0)
        gen.alpha = 1.0

        images = []
        with torch.no_grad():
            z = torch.randn(n_images, self.latent_size, device=self.model_.device)
            out = gen(z)  # (N, 1, D, H, W)

        for i in range(n_images):
            arr = out[i, 0].cpu().numpy()
            if data_type is not None:
                arr = arr.astype(data_type)
            images.append(nib.Nifti1Image(arr, np.eye(4)))

        return images

    def save(self, save_dir: str | Path) -> None:
        """Save Lightning checkpoint + croissant.json."""
        from .croissant import write_model_croissant

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), save_dir / "model.pth")
        write_model_croissant(save_dir, self, self._training_result, self._dataset)

    def _build_model(self) -> nn.Module:
        from nobrainer.models import get as get_model

        return get_model(self.base_model)(**self.model_args)

    def _restore_from_provenance(self, prov: dict) -> None:
        self.base_model = prov.get("model_architecture", "progressivegan")
        self.model_args = prov.get("model_args", {})
        self.latent_size = self.model_args.get("latent_size", 256)
