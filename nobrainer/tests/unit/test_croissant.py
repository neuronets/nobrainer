"""Unit tests for nobrainer.processing.croissant helpers (T024)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np

from nobrainer.processing.croissant import (
    _sha256,
    validate_croissant,
    write_dataset_croissant,
    write_model_croissant,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nifti(shape=(16, 16, 16), tmpdir: Path | None = None) -> str:
    """Write a synthetic NIfTI file and return its path."""
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmpdir / f"vol_{np.random.randint(0, int(1e6))}.nii.gz"
    nib.save(img, str(path))
    return str(path)


def _make_fake_estimator(model_name="unet"):
    """Create a mock estimator with typical attributes."""
    est = MagicMock()
    est.base_model = model_name
    est.model_args = {"channels": (4, 8), "strides": (2,)}
    est.n_classes_ = 2
    est.block_shape_ = (16, 16, 16)
    est._optimizer_class = "Adam"
    est._optimizer_args = {"lr": "0.001"}
    est._loss_name = "CrossEntropyLoss"
    return est


def _make_fake_dataset(tmp_path, n=2):
    """Create a mock dataset with real NIfTI files."""
    data = []
    for _ in range(n):
        img = _make_nifti((16, 16, 16), tmp_path)
        lbl = _make_nifti((16, 16, 16), tmp_path)
        data.append({"image": img, "label": lbl})

    ds = MagicMock()
    ds.data = data
    ds.volume_shape = (16, 16, 16)
    ds.n_classes = 2
    ds._block_shape = (16, 16, 16)
    return ds


# ---------------------------------------------------------------------------
# Tests: write_model_croissant
# ---------------------------------------------------------------------------


class TestWriteModelCroissant:
    def test_creates_valid_jsonld(self, tmp_path):
        """write_model_croissant() creates a valid JSON-LD file."""
        est = _make_fake_estimator()
        ds = _make_fake_dataset(tmp_path)
        result = {
            "history": [
                {"epoch": 1, "loss": 0.5},
                {"epoch": 2, "loss": 0.4},
            ],
            "checkpoint_path": None,
        }
        out = write_model_croissant(tmp_path, est, result, ds)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "@context" in data
        assert "@type" in data
        assert data["@type"] == "sc:Dataset"

    def test_required_provenance_fields(self, tmp_path):
        """Provenance must contain all required fields."""
        est = _make_fake_estimator()
        ds = _make_fake_dataset(tmp_path)
        result = {
            "history": [
                {"epoch": 1, "loss": 0.5},
                {"epoch": 2, "loss": 0.4},
            ],
            "checkpoint_path": None,
        }
        out = write_model_croissant(tmp_path, est, result, ds)
        data = json.loads(out.read_text())
        prov = data["nobrainer:provenance"]
        assert "source_datasets" in prov
        assert "training_date" in prov
        assert "nobrainer_version" in prov
        assert "model_architecture" in prov

    def test_provenance_model_architecture(self, tmp_path):
        est = _make_fake_estimator("meshnet")
        ds = _make_fake_dataset(tmp_path)
        out = write_model_croissant(tmp_path, est, None, ds)
        data = json.loads(out.read_text())
        assert data["nobrainer:provenance"]["model_architecture"] == "meshnet"

    def test_sha256_checksums_for_source_datasets(self, tmp_path):
        """Source datasets must have SHA256 checksums."""
        est = _make_fake_estimator()
        ds = _make_fake_dataset(tmp_path, n=2)
        out = write_model_croissant(tmp_path, est, None, ds)
        data = json.loads(out.read_text())
        sources = data["nobrainer:provenance"]["source_datasets"]
        assert len(sources) >= 1
        for src in sources:
            assert "sha256" in src
            assert len(src["sha256"]) == 64  # SHA256 hex digest length


class TestSHA256:
    def test_checksum_computed(self, tmp_path):
        """_sha256 returns a 64-char hex digest for a file."""
        path = _make_nifti((4, 4, 4), tmp_path)
        digest = _sha256(path)
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_deterministic(self, tmp_path):
        """Same file produces same checksum."""
        path = _make_nifti((4, 4, 4), tmp_path)
        assert _sha256(path) == _sha256(path)


# ---------------------------------------------------------------------------
# Tests: validate_croissant
# ---------------------------------------------------------------------------


class TestValidateCroissant:
    def test_returns_true_on_valid(self, tmp_path):
        """validate_croissant() returns True on a valid file."""
        est = _make_fake_estimator()
        ds = _make_fake_dataset(tmp_path)
        out = write_model_croissant(tmp_path, est, None, ds)
        assert validate_croissant(out) is True


# ---------------------------------------------------------------------------
# Tests: write_dataset_croissant
# ---------------------------------------------------------------------------


class TestWriteDatasetCroissant:
    def test_writes_dataset_metadata(self, tmp_path):
        """write_dataset_croissant() writes a valid JSON-LD."""
        ds = _make_fake_dataset(tmp_path)
        out = write_dataset_croissant(tmp_path / "ds_croissant.json", ds)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "@context" in data
        assert "@type" in data
        assert data["@type"] == "sc:Dataset"

    def test_dataset_info_present(self, tmp_path):
        ds = _make_fake_dataset(tmp_path)
        out = write_dataset_croissant(tmp_path / "ds_croissant.json", ds)
        data = json.loads(out.read_text())
        assert "nobrainer:dataset_info" in data
        info = data["nobrainer:dataset_info"]
        assert info["n_classes"] == 2
        assert info["n_volumes"] == 2

    def test_distribution_has_sha256(self, tmp_path):
        ds = _make_fake_dataset(tmp_path)
        out = write_dataset_croissant(tmp_path / "ds_croissant.json", ds)
        data = json.loads(out.read_text())
        for item in data["distribution"]:
            assert "sha256" in item
            assert len(item["sha256"]) == 64
