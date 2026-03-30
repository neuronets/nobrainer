"""Block-based prediction utilities (PyTorch, no TensorFlow)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from nobrainer.training import get_device


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
        device = get_device()
    device = torch.device(device)

    # Multi-GPU: distribute blocks across GPUs when device="cuda" and >1 GPU
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
    use_multi_gpu = n_gpus > 1

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

    if use_multi_gpu:
        # Replicate model to each GPU
        models = [model.to(torch.device(f"cuda:{i}")) for i in range(n_gpus)]
        for m in models:
            m.eval()
    else:
        model = model.to(device)
        model.eval()

    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            chunk = blocks[start : start + batch_size]  # (B, bD, bH, bW)
            if normalizer is not None:
                chunk = np.stack([normalizer(b) for b in chunk])

            if use_multi_gpu:
                # Round-robin distribute across GPUs
                gpu_idx = (start // batch_size) % n_gpus
                dev = torch.device(f"cuda:{gpu_idx}")
                tensor = torch.from_numpy(chunk[:, None]).to(dev)
                out = models[gpu_idx](tensor)
            else:
                tensor = torch.from_numpy(chunk[:, None]).to(device)
                out = model(tensor)

            if return_labels:
                out = out.argmax(dim=1, keepdim=True).float()
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
    """MC-Dropout / Bayesian uncertainty estimation.

    Runs ``n_samples`` stochastic forward passes with the model in **train**
    mode (activating Dropout and Pyro sampling in Bayesian layers) and
    returns mean label, predictive variance, and predictive entropy maps.

    Parameters
    ----------
    inputs : path, ndarray, or Nifti1Image
        Input brain MRI (same format as :func:`predict`).
    model : nn.Module
        Trained segmentation model.  Should contain dropout or Bayesian
        layers so that repeated forward passes are stochastic.
    n_samples : int
        Number of Monte-Carlo forward passes.
    block_shape, batch_size, device
        Same semantics as :func:`predict`.

    Returns
    -------
    label_img : nib.Nifti1Image
        Mean class label (argmax over mean softmax probabilities).
    variance_img : nib.Nifti1Image
        Mean predictive variance across classes.
    entropy_img : nib.Nifti1Image
        Predictive entropy of the mean softmax distribution.
    """
    if device is None:
        device = get_device()
    device = torch.device(device)

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

    padded, pad = _pad_to_multiple(arr3d, block_shape)
    blocks, grid = _extract_blocks(padded, block_shape)
    n_blocks = blocks.shape[0]

    model = model.to(device)
    # Keep model in train mode so dropout / Pyro sampling remains stochastic
    model.train()

    # Welford's online algorithm: accumulate mean and M2 incrementally
    # so we only keep 2 block-level arrays in memory, not n_samples copies.
    mean_probs: np.ndarray | None = None  # running mean
    m2_probs: np.ndarray | None = None  # running sum of squared deviations

    with torch.no_grad():
        for sample_idx in range(n_samples):
            preds: list[np.ndarray] = []
            for start in range(0, n_blocks, batch_size):
                chunk = blocks[start : start + batch_size]
                tensor = torch.from_numpy(chunk[:, None]).to(device)
                out = model(tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds.append(probs)
            sample = np.concatenate(preds, axis=0)  # (N_blocks, C, bD, bH, bW)

            if mean_probs is None:
                mean_probs = sample.copy()
                m2_probs = np.zeros_like(sample)
            else:
                delta = sample - mean_probs
                mean_probs += delta / (sample_idx + 1)
                delta2 = sample - mean_probs
                m2_probs += delta * delta2

    var_probs = m2_probs / max(n_samples, 1)  # population variance
    n_classes = mean_probs.shape[1]

    # Reconstruct full volumes
    full_mean = _stitch_blocks(
        mean_probs, grid, block_shape, pad, orig_shape, n_classes
    )
    full_var = _stitch_blocks(var_probs, grid, block_shape, pad, orig_shape, n_classes)

    # Mean variance across classes
    mean_var = full_var.mean(axis=0)  # (D, H, W)

    # Predictive entropy: -sum(p * log(p + eps), axis=classes)
    eps = 1e-8
    entropy = -(full_mean * np.log(full_mean + eps)).sum(axis=0)  # (D, H, W)

    # Label = argmax of mean softmax
    if n_classes == 1:
        labels = (full_mean[0] > 0.5).astype(np.float32)
    else:
        labels = full_mean.argmax(axis=0).astype(np.float32)

    label_img = nib.Nifti1Image(labels, affine)
    var_img = nib.Nifti1Image(mean_var.astype(np.float32), affine)
    entropy_img = nib.Nifti1Image(entropy.astype(np.float32), affine)
    return label_img, var_img, entropy_img


__all__ = ["predict", "predict_with_uncertainty"]
