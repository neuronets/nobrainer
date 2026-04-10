"""Unit tests for nobrainer.processing.dataset.Dataset fluent builder (T013)."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader

from nobrainer.processing.dataset import Dataset

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


def _make_file_pairs(n, shape, tmpdir):
    """Create n (image, label) NIfTI file pairs."""
    pairs = []
    for _ in range(n):
        img_path = _make_nifti(shape, tmpdir)
        lbl_path = _make_nifti(shape, tmpdir)
        pairs.append((img_path, lbl_path))
    return pairs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFromFiles:
    def test_tuple_format(self, tmp_path):
        """from_files() accepts list of (image, label) tuples."""
        pairs = _make_file_pairs(3, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=2)
        assert len(ds.data) == 3
        assert all("image" in d and "label" in d for d in ds.data)

    def test_dict_format(self, tmp_path):
        """from_files() accepts list of dicts."""
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        dicts = [{"image": img, "label": lbl} for img, lbl in pairs]
        ds = Dataset.from_files(dicts, block_shape=(16, 16, 16), n_classes=2)
        assert len(ds.data) == 2

    def test_volume_shape_detected(self, tmp_path):
        """from_files() detects volume_shape from the first NIfTI."""
        pairs = _make_file_pairs(1, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        assert ds.volume_shape == (16, 16, 16)


class TestFluentChaining:
    def test_batch_returns_self(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        result = ds.batch(4)
        assert result is ds

    def test_shuffle_returns_self(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        result = ds.shuffle()
        assert result is ds

    def test_augment_returns_self(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        result = ds.augment()
        assert result is ds

    def test_chaining(self, tmp_path):
        """Chaining .batch().shuffle().augment() returns the same instance."""
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        result = ds.batch(2).shuffle().augment()
        assert result is ds
        assert ds._batch_size == 2
        assert ds._shuffle is True
        assert ds._augment is True


class TestSplit:
    def test_split_sizes(self, tmp_path):
        """split() returns two Datasets with correct combined size."""
        pairs = _make_file_pairs(10, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=2)
        train, val = ds.split(eval_size=0.2)
        assert len(train.data) + len(val.data) == 10
        assert len(val.data) == 2  # int(10 * 0.2) = 2

    def test_split_returns_datasets(self, tmp_path):
        pairs = _make_file_pairs(4, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        train, val = ds.split(eval_size=0.25)
        assert isinstance(train, Dataset)
        assert isinstance(val, Dataset)


class TestDataloader:
    def test_returns_dataloader(self, tmp_path):
        """dataloader property returns a torch DataLoader."""
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=2).batch(2)
        loader = ds.dataloader
        assert isinstance(loader, DataLoader)

    def test_batch_produces_data(self, tmp_path):
        """DataLoader yields batches with image data."""
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=2).batch(2)
        batch = next(iter(ds.dataloader))
        # MONAI DataLoader returns dict with "image" key
        assert "image" in batch
        assert batch["image"].ndim == 5  # (B, C, D, H, W)


class TestMetadataProperties:
    def test_batch_size(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16)).batch(4)
        assert ds.batch_size == 4

    def test_block_shape(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        assert ds.block_shape == (16, 16, 16)

    def test_volume_shape(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16))
        assert ds.volume_shape == (16, 16, 16)

    def test_n_classes(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=3)
        assert ds.n_classes == 3


class TestToCroissant:
    def test_writes_valid_jsonld(self, tmp_path):
        """to_croissant() writes valid JSON-LD with @context and fields."""
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=2)
        out = ds.to_croissant(tmp_path / "dataset_croissant.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "@context" in data
        assert "@type" in data
        assert data["@type"] == "sc:Dataset"

    def test_has_dataset_info(self, tmp_path):
        pairs = _make_file_pairs(2, (16, 16, 16), tmp_path)
        ds = Dataset.from_files(pairs, block_shape=(16, 16, 16), n_classes=2)
        out = ds.to_croissant(tmp_path / "dataset_croissant.json")
        data = json.loads(out.read_text())
        assert "nobrainer:dataset_info" in data
        info = data["nobrainer:dataset_info"]
        assert info["n_classes"] == 2
        assert info["n_volumes"] == 2
