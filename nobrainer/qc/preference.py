"""Machine preference scoring via downstream task degradation.

Computes per-structure Dice scores between reference and corrupted
SynthSeg segmentations. The Dice degradation between the two
segmentations serves as a machine-derived preference signal: where the
downstream pipeline's output disagrees most with itself under
corruption is a measure of how much the corruption damaged the inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import torch

logger = logging.getLogger(__name__)

# Key FreeSurfer structures and their label IDs.
#
# Cortex is the union of the coarse FreeSurfer labels (3 lh, 42 rh) AND the
# Desikan-Killiany parcellation labels (1001-1035 lh, 2001-2035 rh). SynthSeg
# with ``--parc`` REPLACES the coarse cortex labels with DK; without ``--parc``
# only 3/42 are emitted. Taking the union makes cortex_dice defined in either
# mode — a seg that has any cortical voxel gets a value, not NaN.
#
# Subcortical structures stay as single-label pairs. On scans with limited
# FOV (e.g. FastMRI axial slabs that miss brainstem/cerebellum) those Dice
# values are NaN, which is correct — absent-from-seg is not a zero-overlap
# failure, it's a "structure not in the image" statement. mean_dice collapses
# to the average over structures that were present on both sides.
_DK_LH_LABELS: list[int] = list(range(1001, 1036))
_DK_RH_LABELS: list[int] = list(range(2001, 2036))

STRUCTURE_LABELS: dict[str, list[int]] = {
    "hippocampus": [17, 53],
    "cortex": [3, 42, *_DK_LH_LABELS, *_DK_RH_LABELS],
    "ventricle": [4, 43],
    "thalamus": [10, 49],
    "caudate": [11, 50],
    "putamen": [12, 51],
    "brainstem": [16],
    "cerebellum": [8, 47],
}


def _dice_for_labels(
    ref: torch.Tensor,
    cor: torch.Tensor,
    label_ids: list[int],
) -> float:
    """Compute Dice for a set of merged label IDs.

    Parameters:
        ref: Reference segmentation (integer labels).
        cor: Corrupted segmentation (integer labels).
        label_ids: FreeSurfer label IDs to merge into a single binary mask.

    Returns:
        Dice coefficient. NaN if the structure is absent in both.
    """
    ref_mask = torch.zeros_like(ref, dtype=torch.bool)
    cor_mask = torch.zeros_like(cor, dtype=torch.bool)
    for lid in label_ids:
        ref_mask = ref_mask | (ref == lid)
        cor_mask = cor_mask | (cor == lid)

    ref_sum = ref_mask.sum()
    cor_sum = cor_mask.sum()

    if ref_sum == 0 and cor_sum == 0:
        return float("nan")

    intersection = (ref_mask & cor_mask).sum().float()
    return (2.0 * intersection / (ref_sum + cor_sum).float()).item()


def compute_dice_preference(
    ref_seg_path: str | Path,
    cor_seg_path: str | Path,
) -> dict[str, float]:
    """Compute per-structure Dice between reference and corrupted segmentations.

    Parameters:
        ref_seg_path: Path to reference SynthSeg segmentation NIfTI.
        cor_seg_path: Path to corrupted SynthSeg segmentation NIfTI.

    Returns:
        Keys: "mean_dice", plus "{structure}_dice" for each structure
        in STRUCTURE_LABELS.
    """
    ref_nii = nib.load(str(ref_seg_path))
    cor_nii = nib.load(str(cor_seg_path))

    ref = torch.from_numpy(ref_nii.get_fdata()).long()
    cor = torch.from_numpy(cor_nii.get_fdata()).long()

    results: dict[str, float] = {}
    dice_values: list[float] = []

    for name, label_ids in STRUCTURE_LABELS.items():
        d = _dice_for_labels(ref, cor, label_ids)
        results[f"{name}_dice"] = d
        if not (d != d):  # not NaN
            dice_values.append(d)

    results["mean_dice"] = (
        sum(dice_values) / len(dice_values) if dice_values else float("nan")
    )

    return results
