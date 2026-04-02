"""Shared utilities for nobrainer models and training."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def unpack_batch(
    batch: dict | list | tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract image and label tensors from a batch, move to device.

    Handles both dict-style (MONAI) and tuple-style (TensorDataset) batches.
    Squeezes label channel dim and casts to long for CrossEntropyLoss.

    Parameters
    ----------
    batch : dict, list, or tuple
        A batch from a DataLoader.
    device : torch.device
        Target device.

    Returns
    -------
    images : torch.Tensor
        Shape ``(B, C, D, H, W)`` on *device*.
    labels : torch.Tensor
        Shape ``(B, D, H, W)`` long dtype on *device*.
    """
    if isinstance(batch, dict):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
    elif isinstance(batch, (list, tuple)):
        images = batch[0].to(device)
        labels = batch[1].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Squeeze channel dim from labels if present
    if labels.ndim == images.ndim and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    # Cast float labels to long for CrossEntropyLoss
    if labels.dtype in (torch.float32, torch.float64):
        labels = labels.long()

    return images, labels


def load_input(
    inputs: str | Path | np.ndarray | nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a 3D volume from various input types.

    Parameters
    ----------
    inputs : str, Path, ndarray, or Nifti1Image
        Input volume.

    Returns
    -------
    arr : np.ndarray
        3D array, shape ``(D, H, W)``.
    affine : np.ndarray or None
        4x4 affine matrix (None if input is raw array).
    """
    if isinstance(inputs, (str, Path)):
        img = nib.load(inputs)
        return np.asarray(img.dataobj, dtype=np.float32), img.affine
    elif isinstance(inputs, nib.Nifti1Image):
        return np.asarray(inputs.dataobj, dtype=np.float32), inputs.affine
    elif isinstance(inputs, np.ndarray):
        return inputs.astype(np.float32), None
    else:
        raise TypeError(f"Unsupported input type: {type(inputs)}")


def model_supports_mc(model: torch.nn.Module) -> bool:
    """Check if a model supports the ``mc`` keyword argument in forward().

    Returns True if the model has a ``supports_mc`` class attribute set
    to True, or if its forward method accepts an ``mc`` parameter.
    """
    if getattr(model, "supports_mc", False):
        return True
    # Check the forward signature
    import inspect

    sig = inspect.signature(model.forward)
    return "mc" in sig.parameters
