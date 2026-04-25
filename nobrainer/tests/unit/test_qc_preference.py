"""Unit tests for nobrainer.qc.preference."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _make_seg(tmp_path: Path, name: str, perturb: bool = False) -> Path:
    """Create a segmentation. If perturb=True, shift some labels."""
    seg = np.zeros((32, 32, 32), dtype=np.int32)
    seg[8:24, 8:24, 8:24] = 3  # cortex
    seg[10:22, 10:22, 10:22] = 2  # WM
    seg[14:18, 14:18, 14:18] = 4  # ventricle
    seg[10:14, 10:14, 10:14] = 17  # L hippocampus

    if perturb:
        # Shift hippocampus region → reduces Dice
        seg[10:14, 10:14, 10:14] = 0
        seg[12:16, 12:16, 12:16] = 17

    path = tmp_path / name
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(path))
    return path


class TestComputeDicePreference:
    def test_identical_segs_perfect_dice(self, tmp_path):
        from nobrainer.qc.preference import compute_dice_preference

        seg = _make_seg(tmp_path, "seg.nii.gz")
        result = compute_dice_preference(seg, seg)
        assert result["mean_dice"] == pytest.approx(1.0)
        assert result["hippocampus_dice"] == pytest.approx(1.0)

    def test_perturbed_lower_dice(self, tmp_path):
        from nobrainer.qc.preference import compute_dice_preference

        ref = _make_seg(tmp_path, "ref.nii.gz", perturb=False)
        cor = _make_seg(tmp_path, "cor.nii.gz", perturb=True)
        result = compute_dice_preference(ref, cor)
        assert result["hippocampus_dice"] < 1.0
        assert result["mean_dice"] < 1.0

    def test_returns_all_structures(self, tmp_path):
        from nobrainer.qc.preference import STRUCTURE_LABELS, compute_dice_preference

        seg = _make_seg(tmp_path, "seg.nii.gz")
        result = compute_dice_preference(seg, seg)
        for name in STRUCTURE_LABELS:
            assert f"{name}_dice" in result
        assert "mean_dice" in result

    def test_absent_structure_is_nan(self, tmp_path):
        """Structures not present in either seg should return NaN."""
        from nobrainer.qc.preference import compute_dice_preference

        # Our _make_seg doesn't include brainstem (label 16)
        seg = _make_seg(tmp_path, "seg.nii.gz")
        result = compute_dice_preference(seg, seg)
        assert result["brainstem_dice"] != result["brainstem_dice"]  # NaN

    def test_cortex_dice_is_one_for_identical(self, tmp_path):
        from nobrainer.qc.preference import compute_dice_preference

        seg = _make_seg(tmp_path, "seg.nii.gz")
        result = compute_dice_preference(seg, seg)
        assert result["cortex_dice"] == pytest.approx(1.0)
