"""Unit tests for nobrainer.processing.generation.Generation estimator (T029)."""

from __future__ import annotations

import json

import nibabel as nib
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.processing.generation import Generation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPATIAL = 4
GAN_ARGS = {
    "latent_size": 8,
    "fmap_base": 16,
    "fmap_max": 16,
    "resolution_schedule": [4],
    "steps_per_phase": 100,
}


class _FakeDataset:
    """Minimal dataset-like object for Generation.fit()."""

    def __init__(self, loader):
        self._loader = loader
        self.data = []

    @property
    def dataloader(self):
        return self._loader


def _make_fake_dataset(n=4, spatial=SPATIAL, batch_size=2):
    """Build a fake dataset with tiny synthetic volumes."""
    imgs = torch.randn(n, 1, spatial, spatial, spatial)
    loader = DataLoader(TensorDataset(imgs), batch_size=batch_size)
    return _FakeDataset(loader)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerationFit:
    def test_fit_returns_self(self):
        """Generation('progressivegan').fit() returns self."""
        ds = _make_fake_dataset()
        gen = Generation("progressivegan", model_args=GAN_ARGS, multi_gpu=False)
        result = gen.fit(
            ds,
            epochs=10,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        assert result is gen

    def test_model_created_after_fit(self):
        ds = _make_fake_dataset()
        gen = Generation("progressivegan", model_args=GAN_ARGS, multi_gpu=False)
        gen.fit(
            ds,
            epochs=5,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        assert gen.model_ is not None


class TestGenerationGenerate:
    def test_generate_returns_list_of_nifti(self):
        """.generate(2) returns list of 2 nibabel.Nifti1Image."""
        ds = _make_fake_dataset()
        gen = Generation("progressivegan", model_args=GAN_ARGS, multi_gpu=False)
        gen.fit(
            ds,
            epochs=5,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        images = gen.generate(2)
        assert isinstance(images, list)
        assert len(images) == 2
        for img in images:
            assert isinstance(img, nib.Nifti1Image)


class TestGenerationSave:
    def test_save_creates_croissant(self, tmp_path):
        """.save() creates croissant.json."""
        ds = _make_fake_dataset()
        gen = Generation("progressivegan", model_args=GAN_ARGS, multi_gpu=False)
        gen.fit(
            ds,
            epochs=5,
            accelerator="cpu",
            enable_progress_bar=False,
        )
        save_dir = tmp_path / "gen_out"
        gen.save(save_dir)
        assert (save_dir / "model.pth").exists()
        assert (save_dir / "croissant.json").exists()
        data = json.loads((save_dir / "croissant.json").read_text())
        assert "@context" in data
        assert "nobrainer:provenance" in data
