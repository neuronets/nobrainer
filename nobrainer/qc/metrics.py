"""Signal-based image quality metric extraction.

Computes mriqc-style IQMs using PyTorch without requiring
the full mriqc BIDS-app pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import torch

logger = logging.getLogger(__name__)

# FreeSurfer label IDs for tissue classification.
#
# GM is the union of the coarse FreeSurfer cortex labels (3 lh, 42 rh) AND
# the Desikan-Killiany parcellation labels (1001-1035 lh, 2001-2035 rh).
# SynthSeg run with ``--parc`` REPLACES the coarse cortex labels with DK;
# without ``--parc`` only 3/42 are emitted. Taking the union keeps CNR and
# CJV defined in either mode — a seg that has any cortical voxel gets a GM
# mask, which both metrics require. Parallel to the cortex-label widening
# in nobrainer.qc.preference.
#
# WM labels are not affected by ``--parc``.
_WM_LABELS = {2, 41}
_DK_LH_LABELS = set(range(1001, 1036))
_DK_RH_LABELS = set(range(2001, 2036))
_GM_LABELS = {3, 42} | _DK_LH_LABELS | _DK_RH_LABELS


def _get_tissue_masks(
    volume: torch.Tensor,
    seg: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Derive brain/background/WM/GM masks.

    Parameters:
        volume: 3D intensity volume.
        seg: SynthSeg segmentation (integer labels). If None, uses Otsu
            thresholding as fallback for brain/background only.

    Returns:
        Boolean masks: "brain", "background", and optionally "wm", "gm".
    """
    masks: dict[str, torch.Tensor] = {}

    if seg is not None:
        masks["brain"] = seg > 0
        masks["background"] = seg == 0
        masks["wm"] = torch.zeros_like(seg, dtype=torch.bool)
        for label in _WM_LABELS:
            masks["wm"] = masks["wm"] | (seg == label)
        masks["gm"] = torch.zeros_like(seg, dtype=torch.bool)
        for label in _GM_LABELS:
            masks["gm"] = masks["gm"] | (seg == label)
        return masks

    # Otsu fallback (pure PyTorch)
    flat = volume[volume > 0].flatten()
    if flat.numel() < 10:
        masks["brain"] = volume > 0
        masks["background"] = volume <= 0
        return masks

    hist = torch.histc(flat, bins=256)
    bin_edges = torch.linspace(flat.min().item(), flat.max().item(), 257)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    cumsum = torch.cumsum(hist, dim=0)
    cumsum_val = torch.cumsum(hist * bin_centers, dim=0)

    w0 = cumsum / total
    w1 = 1 - w0
    mean0 = cumsum_val / (cumsum + 1e-8)
    mean1 = (cumsum_val[-1] - cumsum_val) / (total - cumsum + 1e-8)

    inter_class_var = w0 * w1 * (mean0 - mean1) ** 2
    threshold = bin_centers[torch.argmax(inter_class_var)]

    masks["brain"] = volume > threshold
    masks["background"] = ~masks["brain"]
    return masks


def extract_iqms(
    scan_path: str | Path,
    seg_path: str | Path | None = None,
) -> dict[str, float]:
    """Extract image quality metrics from a single scan.

    Parameters:
        scan_path: Path to NIfTI scan.
        seg_path: Path to SynthSeg segmentation. If provided, used for tissue
            classification. If None, falls back to Otsu thresholding
            (only brain/background available; CNR and CJV will be NaN).

    Returns:
        Keys: "snr", "cnr", "efc", "fber", "cjv".
    """
    nii = nib.load(str(scan_path))
    volume = torch.from_numpy(nii.get_fdata()).float()

    seg = None
    if seg_path is not None:
        seg_nii = nib.load(str(seg_path))
        seg = torch.from_numpy(seg_nii.get_fdata()).long()

    masks = _get_tissue_masks(volume, seg)

    brain = volume[masks["brain"]]
    bg = volume[masks["background"]]

    bg_std = bg.std() if bg.numel() > 1 else torch.tensor(1e-8)

    # SNR: mean(brain) / std(background)
    snr = (brain.mean() / (bg_std + 1e-8)).item()

    # FBER: mean(foreground_energy) / mean(background_energy)
    fg_energy = (brain**2).mean()
    bg_energy = (bg**2).mean() if bg.numel() > 0 else torch.tensor(1e-8)
    fber = (fg_energy / (bg_energy + 1e-8)).item()

    # EFC: entropy focus criterion
    # -sum(p * log(p)) where p = normalized gradient magnitude histogram
    grad_x = torch.diff(volume, dim=0).abs()
    grad_y = torch.diff(volume, dim=1).abs()
    grad_z = torch.diff(volume, dim=2).abs()
    # Use mean gradient magnitude (trim to common shape)
    min_shape = [
        min(grad_x.shape[i], grad_y.shape[i], grad_z.shape[i]) for i in range(3)
    ]
    grad_mag = (
        grad_x[: min_shape[0], : min_shape[1], : min_shape[2]]
        + grad_y[: min_shape[0], : min_shape[1], : min_shape[2]]
        + grad_z[: min_shape[0], : min_shape[1], : min_shape[2]]
    )
    grad_flat = grad_mag.flatten()
    grad_sum = grad_flat.sum() + 1e-8
    p = grad_flat / grad_sum
    p = p[p > 0]
    efc = -(p * p.log()).sum().item()

    # CNR and CJV require WM/GM segmentation
    cnr = float("nan")
    cjv = float("nan")
    if "wm" in masks and "gm" in masks:
        wm = volume[masks["wm"]]
        gm = volume[masks["gm"]]
        if wm.numel() > 1 and gm.numel() > 1:
            cnr = ((wm.mean() - gm.mean()).abs() / (bg_std + 1e-8)).item()
            mean_diff = (wm.mean() - gm.mean()).abs()
            cjv = ((wm.std() + gm.std()) / (mean_diff + 1e-8)).item()

    return {"snr": snr, "cnr": cnr, "efc": efc, "fber": fber, "cjv": cjv}
