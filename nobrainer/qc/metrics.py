"""Signal-based image quality metric extraction.

Computes mriqc-style IQMs (SNR, CNR, EFC, FBER, CJV) using PyTorch
without requiring the full mriqc BIDS-app pipeline.

Performance (2026-04-27 patch):
    Pre-patch ``_get_tissue_masks`` iterated ``_GM_LABELS`` (72 labels under
    ``--parc``) in Python with full-volume OR-merging — same bottleneck
    pattern as :mod:`nobrainer.qc.preference`. The current implementation
    uses :func:`torch.isin` (single C++ vectorised call) and moves volume +
    seg to CUDA when available. Numerics may differ at float32 ULP level
    between CPU and GPU; the algebraic definitions are unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import nibabel.processing
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
# in :mod:`nobrainer.qc.preference`.
#
# WM labels are not affected by ``--parc``.
_WM_LABELS: list[int] = [2, 41]
_DK_LH_LABELS: list[int] = list(range(1001, 1036))
_DK_RH_LABELS: list[int] = list(range(2001, 2036))
_GM_LABELS: list[int] = [3, 42, *_DK_LH_LABELS, *_DK_RH_LABELS]


def _get_tissue_masks(
    volume: torch.Tensor,
    seg: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Derive brain/background/WM/GM masks.

    Vectorised: a single :func:`torch.isin` per tissue class replaces the
    per-label Python OR-loop. Operates on whichever device the input
    tensors live on.

    Parameters:
        volume: 3D intensity volume.
        seg: SynthSeg segmentation (integer labels). If ``None``, uses Otsu
            thresholding as fallback for brain/background only.

    Returns:
        Boolean masks: ``"brain"``, ``"background"``, and (when ``seg`` is
        provided) ``"wm"``, ``"gm"``.
    """
    masks: dict[str, torch.Tensor] = {}

    if seg is not None:
        masks["brain"] = seg > 0
        masks["background"] = seg == 0
        wm_t = torch.tensor(_WM_LABELS, dtype=seg.dtype, device=seg.device)
        gm_t = torch.tensor(_GM_LABELS, dtype=seg.dtype, device=seg.device)
        masks["wm"] = torch.isin(seg, wm_t)
        masks["gm"] = torch.isin(seg, gm_t)
        return masks

    # Otsu fallback (kept identical to pre-patch; runs on whatever device
    # ``volume`` is on).
    flat = volume[volume > 0].flatten()
    if flat.numel() < 10:
        masks["brain"] = volume > 0
        masks["background"] = volume <= 0
        return masks

    hist = torch.histc(flat, bins=256)
    bin_edges = torch.linspace(
        flat.min().item(), flat.max().item(), 257, device=volume.device
    )
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

    Auto-uses CUDA when available — uploads volume + seg once, runs all
    metrics on GPU. Falls back to CPU otherwise.

    Parameters:
        scan_path: Path to NIfTI scan.
        seg_path: Path to SynthSeg segmentation. If provided, used for tissue
            classification. If ``None``, falls back to Otsu thresholding
            (only brain/background available; CNR and CJV will be NaN).

    Returns:
        Keys: ``"snr"``, ``"cnr"``, ``"efc"``, ``"fber"``, ``"cjv"``.
    """
    nii = nib.load(str(scan_path))

    seg_nii = None
    if seg_path is not None:
        seg_nii = nib.load(str(seg_path))
        # SynthSeg writes at 1 mm isotropic even when the input scan is
        # anisotropic (e.g. FastMRI 0.6875 x 0.6875 x 5 mm), so seg and scan
        # frequently have different shapes. Resample the seg into the scan's
        # space with nearest-neighbour so downstream mask indexing works and
        # scan intensity statistics (SNR, CNR, FBER, CJV) are computed at
        # native resolution without interpolation smoothing the noise floor.
        if seg_nii.shape != nii.shape:
            seg_nii = nibabel.processing.resample_from_to(seg_nii, nii, order=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    volume = torch.from_numpy(nii.get_fdata()).float().to(device)

    seg = None
    if seg_nii is not None:
        seg = torch.from_numpy(seg_nii.get_fdata()).long().to(device)

    masks = _get_tissue_masks(volume, seg)

    brain = volume[masks["brain"]]
    bg = volume[masks["background"]]

    bg_std = bg.std() if bg.numel() > 1 else torch.tensor(1e-8, device=device)

    # SNR: mean(brain) / std(background)
    snr = (brain.mean() / (bg_std + 1e-8)).item()

    # FBER: mean(foreground_energy) / mean(background_energy)
    fg_energy = (brain**2).mean()
    bg_energy = (bg**2).mean() if bg.numel() > 0 else torch.tensor(1e-8, device=device)
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
            mean_diff = (wm.mean() - gm.mean()).abs()
            cnr = (mean_diff / (bg_std + 1e-8)).item()
            cjv = ((wm.std() + gm.std()) / (mean_diff + 1e-8)).item()

    return {"snr": snr, "cnr": cnr, "efc": efc, "fber": fber, "cjv": cjv}
