"""Base estimator with Croissant-ML metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseEstimator:
    """Base class for all nobrainer estimators.

    Provides ``save()`` / ``load()`` with Croissant-ML JSON-LD metadata,
    and optional multi-GPU support via DDP.
    """

    state_variables: list[str] = []
    model_: nn.Module | None = None
    _training_result: dict | None = None
    _dataset: Any = None

    def __init__(
        self,
        checkpoint_filepath: str | Path | None = None,
        multi_gpu: bool = True,
    ):
        self.checkpoint_filepath = checkpoint_filepath
        self.multi_gpu = multi_gpu

    @property
    def model(self) -> nn.Module:
        if self.model_ is None:
            raise RuntimeError("Model not trained. Call .fit() first.")
        return self.model_

    def save(self, save_dir: str | Path) -> None:
        """Save model.pth + croissant.json to directory."""
        from .croissant import write_model_croissant

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), save_dir / "model.pth")
        write_model_croissant(save_dir, self, self._training_result, self._dataset)

    @classmethod
    def load(cls, model_dir: str | Path, multi_gpu: bool = True) -> "BaseEstimator":
        """Load estimator from directory with croissant.json metadata."""
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "croissant.json").read_text())
        prov = metadata.get("nobrainer:provenance", {})

        est = cls.__new__(cls)
        est.multi_gpu = multi_gpu
        est.checkpoint_filepath = None
        est._training_result = None
        est._dataset = None

        # Subclass-specific reconstruction
        est._restore_from_provenance(prov)
        est.model_ = est._build_model()
        est.model_.load_state_dict(
            torch.load(model_dir / "model.pth", weights_only=True)
        )
        return est

    def _build_model(self) -> nn.Module:
        """Reconstruct model architecture. Override in subclasses."""
        raise NotImplementedError

    def _restore_from_provenance(self, prov: dict) -> None:
        """Restore state from provenance dict. Override in subclasses."""
        raise NotImplementedError
