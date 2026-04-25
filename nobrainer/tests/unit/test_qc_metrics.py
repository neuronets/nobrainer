"""Unit tests for nobrainer.qc.metrics."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def _make_scan(tmp_path: Path, shape=(32, 32, 32)) -> Path:
    """Create a scan with brain-like contrast (brain center, zero background)."""
    arr = np.zeros(shape, dtype=np.float32)
    # Brain region in center
    arr[8:24, 8:24, 8:24] = np.random.RandomState(0).normal(150, 20, (16, 16, 16))
    arr = np.clip(arr, 0, 300)
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return path


def _make_seg(tmp_path: Path, shape=(32, 32, 32)) -> Path:
    """Create a segmentation with WM (2), GM (3), CSF (4)."""
    seg = np.zeros(shape, dtype=np.int32)
    seg[10:22, 10:22, 10:22] = 2  # WM
    seg[8:24, 8:24, 8:24][seg[8:24, 8:24, 8:24] == 0] = 3  # GM shell
    seg[14:18, 14:18, 14:18] = 4  # CSF center
    path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(path))
    return path


class TestExtractIqms:
    def test_returns_all_keys(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        result = extract_iqms(scan)
        assert set(result.keys()) == {"snr", "cnr", "efc", "fber", "cjv"}

    def test_snr_positive(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        result = extract_iqms(scan)
        assert result["snr"] > 0

    def test_with_segmentation_enables_cnr(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        seg = _make_seg(tmp_path)
        result = extract_iqms(scan, seg_path=seg)
        assert result["cnr"] == result["cnr"]  # not NaN

    def test_without_segmentation_cnr_is_nan(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        result = extract_iqms(scan, seg_path=None)
        assert result["cnr"] != result["cnr"]  # NaN

    def test_efc_is_finite(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        result = extract_iqms(scan)
        assert np.isfinite(result["efc"])

    def test_fber_positive(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        result = extract_iqms(scan)
        assert result["fber"] > 0

    def test_cjv_with_segmentation_is_finite(self, tmp_path):
        from nobrainer.qc.metrics import extract_iqms

        scan = _make_scan(tmp_path)
        seg = _make_seg(tmp_path)
        result = extract_iqms(scan, seg_path=seg)
        assert np.isfinite(result["cjv"])
