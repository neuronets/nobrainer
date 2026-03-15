"""Training utilities with optional multi-GPU DDP support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def fit(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 10,
    gpus: int = 1,
    checkpoint_dir: str | Path | None = None,
    callbacks: list[Any] | None = None,
) -> dict:
    """Train a model with optional multi-GPU DDP.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    loader : DataLoader
        Training data loader.
    criterion : nn.Module
        Loss function.
    optimizer : Optimizer
        PyTorch optimizer.
    max_epochs : int
        Number of training epochs.
    gpus : int
        Number of GPUs to use (1 = single GPU, >1 = DDP).
    checkpoint_dir : path or None
        Directory for saving checkpoints. None disables checkpointing.
    callbacks : list or None
        Optional callback functions called after each epoch.

    Returns
    -------
    dict with keys: final_loss, best_loss, epochs_completed, checkpoint_path
    """
    raise NotImplementedError("fit() will be implemented in T030/T031")
