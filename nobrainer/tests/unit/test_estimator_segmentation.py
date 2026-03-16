"""Unit tests for nobrainer.processing.segmentation.Segmentation estimator (T023)."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.processing.segmentation import Segmentation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPATIAL = 16
N_CLASSES = 2


def _make_nifti(shape=(16, 16, 16), tmpdir: Path | None = None) -> str:
    """Write a synthetic NIfTI file and return its path."""
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmpdir / f"vol_{np.random.randint(0, int(1e6))}.nii.gz"
    nib.save(img, str(path))
    return str(path)


def _make_tiny_loader(n=4, spatial=SPATIAL, n_classes=N_CLASSES, batch_size=2):
    """Create a tiny DataLoader with tuple batches for training."""
    x = torch.randn(n, 1, spatial, spatial, spatial)
    y = torch.randint(0, n_classes, (n, spatial, spatial, spatial))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


class _FakeDataset:
    """Minimal object mimicking the Dataset builder for Segmentation.fit()."""

    def __init__(self, loader, block_shape, volume_shape, n_classes):
        self._loader = loader
        self._block_shape = block_shape
        self.volume_shape = volume_shape
        self.n_classes = n_classes

    @property
    def block_shape(self):
        return self._block_shape

    @property
    def dataloader(self):
        return self._loader


def _make_fake_dataset(n=4, spatial=SPATIAL, n_classes=N_CLASSES, batch_size=2):
    """Build a FakeDataset with a tiny DataLoader."""
    loader = _make_tiny_loader(n, spatial, n_classes, batch_size)
    return _FakeDataset(
        loader,
        block_shape=(spatial, spatial, spatial),
        volume_shape=(spatial, spatial, spatial),
        n_classes=n_classes,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSegmentationFit:
    def test_fit_returns_self(self):
        """Segmentation('unet').fit(ds, epochs=2) returns self."""
        ds = _make_fake_dataset()
        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
            multi_gpu=False,
        )
        result = seg.fit(ds, epochs=2)
        assert result is seg

    def test_model_created_after_fit(self):
        ds = _make_fake_dataset()
        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
            multi_gpu=False,
        )
        seg.fit(ds, epochs=1)
        assert seg.model_ is not None
        assert isinstance(seg.model_, nn.Module)


class TestSegmentationPredict:
    def test_predict_returns_nifti(self, tmp_path):
        """.predict() returns nibabel.Nifti1Image with correct shape."""
        ds = _make_fake_dataset()
        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
            multi_gpu=False,
        )
        seg.fit(ds, epochs=1)

        # Create a test volume
        vol_path = _make_nifti((SPATIAL, SPATIAL, SPATIAL), tmp_path)
        result = seg.predict(vol_path, block_shape=(SPATIAL, SPATIAL, SPATIAL))
        assert isinstance(result, nib.Nifti1Image)
        assert result.shape[:3] == (SPATIAL, SPATIAL, SPATIAL)


class TestSegmentationSaveLoad:
    def test_save_creates_files(self, tmp_path):
        """.save() creates model.pth and croissant.json."""
        ds = _make_fake_dataset()
        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
            multi_gpu=False,
        )
        seg.fit(ds, epochs=1)
        save_dir = tmp_path / "model_out"
        seg.save(save_dir)
        assert (save_dir / "model.pth").exists()
        assert (save_dir / "croissant.json").exists()

    def test_croissant_provenance_fields(self, tmp_path):
        """croissant.json contains all provenance fields."""
        ds = _make_fake_dataset()
        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
            multi_gpu=False,
        )
        seg.fit(ds, epochs=1)
        save_dir = tmp_path / "model_out"
        seg.save(save_dir)
        data = json.loads((save_dir / "croissant.json").read_text())
        prov = data["nobrainer:provenance"]
        assert "source_datasets" in prov
        assert "training_date" in prov
        assert "nobrainer_version" in prov
        assert "model_architecture" in prov
        assert prov["model_architecture"] == "unet"

    def test_load_roundtrip(self, tmp_path):
        """.load() round-trip produces same prediction output."""
        ds = _make_fake_dataset()
        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
            multi_gpu=False,
        )
        seg.fit(ds, epochs=1)

        # Get prediction before save
        test_vol = np.random.rand(SPATIAL, SPATIAL, SPATIAL).astype(np.float32)
        pred_before = seg.predict(test_vol, block_shape=(SPATIAL, SPATIAL, SPATIAL))

        # Save and reload
        save_dir = tmp_path / "model_out"
        seg.save(save_dir)
        loaded = Segmentation.load(save_dir, multi_gpu=False)

        # Predict again
        pred_after = loaded.predict(test_vol, block_shape=(SPATIAL, SPATIAL, SPATIAL))
        np.testing.assert_array_equal(
            np.asarray(pred_before.dataobj),
            np.asarray(pred_after.dataobj),
        )
