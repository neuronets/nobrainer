"""Unit tests for nobrainer.qc.corrupt."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _make_nifti(tmp_path: Path, name: str = "test.nii.gz", shape=(32, 32, 32)) -> Path:
    """Create a synthetic NIfTI file with non-zero brain-like data."""
    arr = np.random.RandomState(42).normal(100, 20, shape).astype(np.float32)
    arr = np.clip(arr, 0, 200)
    path = tmp_path / name
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
    return path


class TestGenerateCorruptedScan:
    def test_produces_output_file(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_scan
        from nobrainer.qc.corruption_configs import get_corruption_configs

        input_path = _make_nifti(tmp_path, "ref.nii.gz")
        output_path = tmp_path / "corrupted" / "out.nii.gz"
        config = get_corruption_configs()["noise"]

        meta = generate_corrupted_scan(
            input_path, output_path, config, severity=3, seed=42
        )

        assert output_path.exists()
        assert output_path.with_suffix(".json").exists()
        assert meta["corruption_type"] == "noise"
        assert meta["severity"] == 3

    def test_preserves_affine(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_scan
        from nobrainer.qc.corruption_configs import get_corruption_configs

        input_path = _make_nifti(tmp_path)
        output_path = tmp_path / "out.nii.gz"
        config = get_corruption_configs()["blur"]

        generate_corrupted_scan(input_path, output_path, config, severity=1, seed=0)

        orig = nib.load(str(input_path))
        corrupted = nib.load(str(output_path))
        np.testing.assert_array_almost_equal(orig.affine, corrupted.affine)

    def test_same_seed_same_output(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_scan
        from nobrainer.qc.corruption_configs import get_corruption_configs

        input_path = _make_nifti(tmp_path)
        config = get_corruption_configs()["noise"]

        out1 = tmp_path / "out1.nii.gz"
        out2 = tmp_path / "out2.nii.gz"

        generate_corrupted_scan(input_path, out1, config, severity=2, seed=123)
        generate_corrupted_scan(input_path, out2, config, severity=2, seed=123)

        data1 = nib.load(str(out1)).get_fdata()
        data2 = nib.load(str(out2)).get_fdata()
        np.testing.assert_array_equal(data1, data2)

    def test_different_severity_different_output(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_scan
        from nobrainer.qc.corruption_configs import get_corruption_configs

        input_path = _make_nifti(tmp_path)
        config = get_corruption_configs()["noise"]

        out1 = tmp_path / "s1.nii.gz"
        out5 = tmp_path / "s5.nii.gz"

        generate_corrupted_scan(input_path, out1, config, severity=1, seed=42)
        generate_corrupted_scan(input_path, out5, config, severity=5, seed=42)

        data1 = nib.load(str(out1)).get_fdata()
        data5 = nib.load(str(out5)).get_fdata()
        # Higher severity should produce larger deviation from original
        orig = nib.load(str(input_path)).get_fdata()
        diff1 = np.abs(data1 - orig).mean()
        diff5 = np.abs(data5 - orig).mean()
        assert diff5 > diff1

    def test_sidecar_json_valid(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_scan
        from nobrainer.qc.corruption_configs import get_corruption_configs

        input_path = _make_nifti(tmp_path)
        output_path = tmp_path / "out.nii.gz"
        config = get_corruption_configs()["gamma"]

        generate_corrupted_scan(input_path, output_path, config, severity=2, seed=0)

        sidecar = json.loads(output_path.with_suffix(".json").read_text())
        assert sidecar["corruption_type"] == "gamma"
        assert sidecar["severity"] == 2
        assert "seed" in sidecar


class TestGenerateCorruptedDataset:
    def test_resume_skips_existing(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_dataset

        input_dir = tmp_path / "refs"
        input_dir.mkdir()
        _make_nifti(input_dir, "scan1.nii.gz")

        output_dir = tmp_path / "corrupted"
        # First run
        meta1 = generate_corrupted_dataset(
            input_dir, output_dir, corruptions=["noise"], severities=[1]
        )
        # Second run (should skip)
        meta2 = generate_corrupted_dataset(
            input_dir, output_dir, corruptions=["noise"], severities=[1]
        )
        assert len(meta1) == 1
        assert len(meta2) == 0  # all skipped

    def test_empty_dir_raises(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_dataset

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            generate_corrupted_dataset(empty, tmp_path / "out")

    def test_unknown_corruption_raises(self, tmp_path):
        from nobrainer.qc.corrupt import generate_corrupted_dataset

        input_dir = tmp_path / "refs"
        input_dir.mkdir()
        _make_nifti(input_dir)
        with pytest.raises(ValueError, match="Unknown corruptions"):
            generate_corrupted_dataset(
                input_dir, tmp_path / "out", corruptions=["nonexistent"]
            )
