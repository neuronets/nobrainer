"""Unit tests for nobrainer.qc.slice_extractor."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def _make_volume(tmp_path: Path, shape=(32, 40, 48)) -> Path:
    arr = np.random.RandomState(0).normal(100, 30, shape).astype(np.float32)
    arr[:5, :, :] = 0  # empty border
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return path


class TestExtractSlices:
    def test_mid_returns_three_orientations(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path)
        slices = extract_slices(vol, method="mid")
        assert len(slices) == 3
        assert "mid_axial" in slices
        assert "mid_coronal" in slices
        assert "mid_sagittal" in slices

    def test_slice_shapes(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path, shape=(32, 40, 48))
        slices = extract_slices(vol, method="mid")
        # Axial slices along dim 2 → shape (32, 40)
        assert slices["mid_axial"].shape == (32, 40)
        # Coronal along dim 1 → shape (32, 48)
        assert slices["mid_coronal"].shape == (32, 48)
        # Sagittal along dim 0 → shape (40, 48)
        assert slices["mid_sagittal"].shape == (40, 48)

    def test_uint8_range(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path)
        slices = extract_slices(vol, method="mid")
        for s in slices.values():
            assert s.dtype == torch.uint8
            assert s.max() <= 255
            assert s.min() >= 0

    def test_max_info_selects_non_empty(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path)
        slices = extract_slices(vol, method="max_info", orientations=["sagittal"])
        # Should NOT select slice 0-4 (all zeros)
        # max_info sagittal = index along dim 0 with most non-zero voxels
        assert "max_info_sagittal" in slices

    def test_unknown_method_raises(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path)
        with pytest.raises(ValueError, match="Unknown method"):
            extract_slices(vol, method="invalid")

    def test_unknown_orientation_raises(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path)
        with pytest.raises(ValueError, match="Unknown orientation"):
            extract_slices(vol, orientations=["diagonal"])

    def test_single_orientation(self, tmp_path):
        from nobrainer.qc.slice_extractor import extract_slices

        vol = _make_volume(tmp_path)
        slices = extract_slices(vol, orientations=["axial"])
        assert len(slices) == 1
        assert "mid_axial" in slices
