"""Block-based prediction utilities (PyTorch, no TensorFlow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


def _pad_to_multiple(
    arr: np.ndarray, block_shape: tuple[int, int, int]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Pad spatial dims of ``arr`` (D, H, W) so each is divisible by block_shape."""
    pads = []
    for dim, bs in zip(arr.shape, block_shape):
        rem = (-dim) % bs
        pads.append((0, rem))
    return np.pad(arr, pads, mode="constant"), tuple(p[1] for p in pads)


def _extract_blocks(arr: np.ndarray, block_shape: tuple[int, int, int]) -> np.ndarray:
    """Split ``arr`` (D, H, W) into non-overlapping blocks of ``block_shape``."""
    D, H, W = arr.shape
    bd, bh, bw = block_shape
    blocks = arr.reshape(D // bd, bd, H // bh, bh, W // bw, bw)
    # → (nd, bD, nh, bH, nw, bW)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5)
    # → (nd, nh, nw, bD, bH, bW)
    nd, nh, nw = D // bd, H // bh, W // bw
    return blocks.reshape(nd * nh * nw, bd, bh, bw), (nd, nh, nw)


def _stitch_blocks(
    block_preds: np.ndarray,
    grid: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    pad: tuple[int, int, int],
    orig_shape: tuple[int, int, int],
    n_classes: int,
) -> np.ndarray:
    """Reconstruct full prediction volume from per-block predictions."""
    nd, nh, nw = grid
    bd, bh, bw = block_shape
    # block_preds: (N_blocks, n_classes, bD, bH, bW)
    vol = block_preds.reshape(nd, nh, nw, n_classes, bd, bh, bw)
    vol = vol.transpose(3, 0, 3, 1, 4, 2, 5)
    # → (n_classes, nd, bD, nh, bH, nw, bW)
    # This ordering is tricky; use a clean loop instead for clarity
    full = np.zeros((n_classes, nd * bd, nh * bh, nw * bw), dtype=block_preds.dtype)
    idx = 0
    for i in range(nd):
        for j in range(nh):
            for k in range(nw):
                full[
                    :,
                    i * bd : (i + 1) * bd,
                    j * bh : (j + 1) * bh,
                    k * bw : (k + 1) * bw,
                ] = block_preds[idx]
                idx += 1

    # Remove padding
    D, H, W = orig_shape
    pd, ph, pw = pad
    end_d = full.shape[1] - pd if pd > 0 else full.shape[1]
    end_h = full.shape[2] - ph if ph > 0 else full.shape[2]
    end_w = full.shape[3] - pw if pw > 0 else full.shape[3]
    return full[:, :end_d, :end_h, :end_w]


def predict(
    inputs: str | Path | np.ndarray | nib.Nifti1Image,
    model: nn.Module,
    block_shape: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 4,
    device: str | torch.device | None = None,
    return_labels: bool = True,
    normalizer: Any | None = None,
) -> nib.Nifti1Image:
    """Run block-based inference on a 3-D brain volume.

    Parameters
    ----------
    inputs : path, ndarray, or Nifti1Image
        Input brain MRI.  If a file path is given, it is loaded with
        nibabel.  If an ndarray, shape must be ``(D, H, W)``.
    model : nn.Module
        Trained PyTorch segmentation model.  Must accept tensors of
        shape ``(N, 1, bD, bH, bW)`` and return ``(N, C, bD, bH, bW)``.
    block_shape : tuple
        Spatial block size ``(bD, bH, bW)`` for patch-based inference.
    batch_size : int
        Number of blocks to process in one forward pass.
    device : str, device, or None
        Compute device.  Defaults to CUDA if available, else CPU.
    return_labels : bool
        If ``True``, return argmax labels.  If ``False``, return class
        probabilities (softmax) as a 4-D volume.
    normalizer : callable or None
        Optional function ``normalizer(arr) → arr`` applied to each block
        before inference.

    Returns
    -------
    nib.Nifti1Image
        Segmentation (or probability) volume with the same affine as the
        input NIfTI.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load input
    affine = np.eye(4)
    if isinstance(inputs, (str, Path)):
        img = nib.load(str(inputs))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine
    elif isinstance(inputs, nib.Nifti1Image):
        arr = np.asarray(inputs.dataobj, dtype=np.float32)
        affine = inputs.affine
    else:
        arr = np.asarray(inputs, dtype=np.float32)

    orig_shape = arr.shape[:3]
    arr3d = arr if arr.ndim == 3 else arr[..., 0]

    # Pad to block-divisible size
    padded, pad = _pad_to_multiple(arr3d, block_shape)
    blocks, grid = _extract_blocks(padded, block_shape)  # (N_blocks, bD, bH, bW)
    n_blocks = blocks.shape[0]

    model = model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            chunk = blocks[start : start + batch_size]  # (B, bD, bH, bW)
            if normalizer is not None:
                chunk = np.stack([normalizer(b) for b in chunk])
            tensor = torch.from_numpy(chunk[:, None]).to(device)  # (B, 1, bD, bH, bW)
            out = model(tensor)  # (B, C, bD, bH, bW)
            if return_labels:
                out = out.argmax(dim=1, keepdim=True).float()  # (B, 1, bD, bH, bW)
            else:
                out = torch.softmax(out, dim=1)
            all_preds.append(out.cpu().numpy())

    block_preds = np.concatenate(all_preds, axis=0)  # (N_blocks, C, bD, bH, bW)
    n_classes = block_preds.shape[1]
    full_pred = _stitch_blocks(
        block_preds, grid, block_shape, pad, orig_shape, n_classes
    )

    # Squeeze class dim for single-class output
    if n_classes == 1:
        spatial = full_pred[0]
    else:
        spatial = full_pred  # (C, D, H, W)

    out_img = nib.Nifti1Image(spatial.astype(np.float32), affine)
    return out_img


def predict_with_uncertainty(
    inputs: str | Path | np.ndarray | nib.Nifti1Image,
    model: nn.Module,
    n_samples: int = 10,
    block_shape: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 4,
    device: str | torch.device | None = None,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image]:
    """MC-Dropout uncertainty estimation.

    Runs ``n_samples`` stochastic forward passes (model in train mode to
    activate dropout) and returns label, variance, and entropy maps.

    .. note::
        Full implementation wired to Bayesian models is in Phase 4 (US2).
        This stub is importable from Phase 3 onward.

    Returns
    -------
    label_img, variance_img, entropy_img : nib.Nifti1Image
    """
    raise NotImplementedError(
        "predict_with_uncertainty() is fully implemented in Phase 4 (US2). "
        "Use predict() for deterministic inference."
    )


__all__ = ["predict", "predict_with_uncertainty"]
