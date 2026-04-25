"""3D → 2D slice extraction strategies for VLM evaluation.

Extracts representative 2D slices from 3D brain MRI volumes
for input to 2D vision-language models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import torch

logger = logging.getLogger(__name__)


def _normalize_to_uint8(slice_2d: torch.Tensor) -> torch.Tensor:
    """Clip to [p2, p98] and normalize to [0, 255]."""
    flat = slice_2d.flatten()
    if flat.numel() == 0:
        return slice_2d.byte()
    p2 = torch.quantile(flat.float(), 0.02)
    p98 = torch.quantile(flat.float(), 0.98)
    clipped = slice_2d.float().clamp(p2, p98)
    if p98 - p2 < 1e-8:
        return torch.zeros_like(clipped).byte()
    normalized = ((clipped - p2) / (p98 - p2) * 255).byte()
    return normalized


def extract_slices(
    scan_path: str | Path,
    method: str = "mid",
    orientations: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract 2D slices from a 3D volume.

    Parameters:
        scan_path: Path to NIfTI volume.
        method: Extraction strategy: "mid" (center slices) or "max_info"
            (slices with maximum non-zero voxel count).
        orientations: Which orientations to extract.
            Default: ["axial", "coronal", "sagittal"].

    Returns:
        Keys: "{method}_{orientation}", values: uint8 tensors (H, W).
    """
    if orientations is None:
        orientations = ["axial", "coronal", "sagittal"]

    nii = nib.load(str(scan_path))
    volume = torch.from_numpy(nii.get_fdata()).float()

    # Axis mapping: orientation → dimension to slice along
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}

    results: dict[str, torch.Tensor] = {}

    for orient in orientations:
        if orient not in axis_map:
            raise ValueError(f"Unknown orientation '{orient}'. Use: {list(axis_map)}")

        axis = axis_map[orient]

        if method == "mid":
            idx = volume.shape[axis] // 2
        elif method == "max_info":
            # Select slice with maximum non-zero voxel count
            counts = (volume > 0).sum(dim=[d for d in range(3) if d != axis])
            idx = counts.argmax().item()
        else:
            raise ValueError(f"Unknown method '{method}'. Use: 'mid', 'max_info'")

        slice_2d = volume.select(axis, idx)
        results[f"{method}_{orient}"] = _normalize_to_uint8(slice_2d)

    return results
