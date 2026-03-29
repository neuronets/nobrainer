"""Unit tests for nobrainer.datasets.openneuro."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestWriteManifest:
    """Test write_manifest without DataLad."""

    def test_creates_csv(self, tmp_path):
        from nobrainer.datasets.openneuro import write_manifest

        pairs = [
            {
                "subject_id": f"sub-{i:02d}",
                "t1w_path": f"/t1_{i}.nii.gz",
                "label_path": f"/lbl_{i}.nii.gz",
            }
            for i in range(5)
        ]
        csv_path = write_manifest(pairs, tmp_path / "manifest.csv")
        assert csv_path.exists()

        import csv

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5
        splits = {r["split"] for r in rows}
        assert splits <= {"train", "val", "test"}

    def test_split_ratios(self, tmp_path):
        from nobrainer.datasets.openneuro import write_manifest

        pairs = [
            {
                "subject_id": f"sub-{i:02d}",
                "t1w_path": f"/t1_{i}.nii.gz",
                "label_path": f"/lbl_{i}.nii.gz",
            }
            for i in range(10)
        ]
        write_manifest(pairs, tmp_path / "m.csv", split_ratios=(60, 20, 20))

        import csv

        with open(tmp_path / "m.csv") as f:
            rows = list(csv.DictReader(f))
        n_train = sum(1 for r in rows if r["split"] == "train")
        assert n_train == 6  # 60% of 10

    def test_dataset_id_column(self, tmp_path):
        from nobrainer.datasets.openneuro import write_manifest

        pairs = [
            {
                "subject_id": "sub-01",
                "dataset_id": "ds000114",
                "t1w_path": "/t1.nii.gz",
                "label_path": "/lbl.nii.gz",
            }
        ]
        csv_path = write_manifest(pairs, tmp_path / "m.csv")

        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["dataset_id"] == "ds000114"


class TestGlobDataset:
    """Test glob_dataset (no DataLad needed)."""

    def test_finds_files(self, tmp_path):
        from nobrainer.datasets.openneuro import glob_dataset

        (tmp_path / "sub-01" / "anat").mkdir(parents=True)
        (tmp_path / "sub-01" / "anat" / "sub-01_T1w.nii.gz").touch()
        (tmp_path / "sub-02" / "anat").mkdir(parents=True)
        (tmp_path / "sub-02" / "anat" / "sub-02_T1w.nii.gz").touch()

        files = glob_dataset(tmp_path, "sub-*/anat/*_T1w.nii.gz")
        assert len(files) == 2

    def test_no_matches(self, tmp_path):
        from nobrainer.datasets.openneuro import glob_dataset

        files = glob_dataset(tmp_path, "sub-*/anat/*_T1w.nii.gz")
        assert files == []


class TestExtractSubjectId:
    def test_from_bids_path(self, tmp_path):
        from nobrainer.datasets.openneuro import _extract_subject_id

        p = tmp_path / "sub-03" / "anat" / "sub-03_T1w.nii.gz"
        assert _extract_subject_id(p) == "sub-03"

    def test_from_filename(self):
        from nobrainer.datasets.openneuro import _extract_subject_id

        p = Path("sub-99_desc-preproc_T1w.nii.gz")
        assert _extract_subject_id(p) == "sub-99"


class TestFileOk:
    def test_real_file(self, tmp_path):
        from nobrainer.datasets.openneuro import _file_ok

        f = tmp_path / "real.nii.gz"
        f.write_bytes(b"data")
        assert _file_ok(f)

    def test_empty_file(self, tmp_path):
        from nobrainer.datasets.openneuro import _file_ok

        f = tmp_path / "empty.nii.gz"
        f.touch()
        assert not _file_ok(f)

    def test_missing_file(self, tmp_path):
        from nobrainer.datasets.openneuro import _file_ok

        assert not _file_ok(tmp_path / "missing.nii.gz")


class TestImportGuard:
    """Test that missing datalad gives a clear error."""

    def test_install_without_datalad(self):
        from nobrainer.datasets.openneuro import install_dataset

        with patch.dict("sys.modules", {"datalad": None, "datalad.api": None}):
            with pytest.raises(ImportError, match="DataLad"):
                install_dataset("ds000114", "/tmp/test")
