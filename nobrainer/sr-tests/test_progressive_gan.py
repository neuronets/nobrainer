"""SR-test: Progressive GAN training smoke test with OME-Zarr pyramidal store.

T028 — Creates a tiny pyramidal store from synthetic data and runs
the GAN for a few steps to verify end-to-end correctness.
Feeds whole volumes (not patches) at resolutions matching the GAN stages.
"""

from __future__ import annotations

from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
import torch


@pytest.fixture()
def tiny_zarr_store(tmp_path):
    """Create a tiny OME-Zarr pyramidal store (5 volumes, 16³, 3 levels).

    Levels: 0→16³, 1→8³, 2→4³.
    """
    from nobrainer.datasets.zarr_store import create_zarr_store

    pairs = []
    for i in range(5):
        img = np.random.randn(16, 16, 16).astype(np.float32)
        lbl = np.random.randint(0, 3, (16, 16, 16), dtype=np.int32)
        affine = np.eye(4)
        img_path = tmp_path / f"sub-{i:02d}_img.nii.gz"
        lbl_path = tmp_path / f"sub-{i:02d}_lbl.nii.gz"
        nib.save(nib.Nifti1Image(img, affine), str(img_path))
        nib.save(nib.Nifti1Image(lbl, affine), str(lbl_path))
        pairs.append((str(img_path), str(lbl_path)))

    store_path = create_zarr_store(
        pairs,
        tmp_path / "test.ome.zarr",
        conform=False,
        levels=3,  # 16³, 8³, 4³
    )
    return store_path


class _WholeVolumeDataset(torch.utils.data.Dataset):
    """Read whole volumes from a zarr store at a given pyramid level."""

    def __init__(self, store_path, level):
        import zarr

        self.store = zarr.open_group(str(store_path), mode="r")
        self.img_arr = self.store[f"images/{level}"]
        self.n = self.img_arr.shape[0]
        # Check for bfloat16
        self._is_bf16 = self.img_arr.attrs.get("_nobrainer_dtype") == "bfloat16"

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        raw = np.asarray(self.img_arr[idx])
        if self._is_bf16:
            from nobrainer.datasets.zarr_store import decode_bfloat16

            raw = decode_bfloat16(raw)
        img = torch.from_numpy(raw.astype(np.float32)).unsqueeze(0)
        return {"image": img}


class TestProgressiveGANSmoke:
    """Smoke test for GAN training with pyramidal OME-Zarr data."""

    def test_training_whole_volumes(self, tiny_zarr_store):
        """GAN trains on whole 4³ volumes from pyramid level 2."""
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader

        from nobrainer.models.generative.progressivegan import ProgressiveGAN

        # Level 2 of 16³ base → 4³ volumes (whole, no patches)
        dataset = _WholeVolumeDataset(tiny_zarr_store, level=2)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = ProgressiveGAN(
            latent_size=64,
            fmap_base=128,
            fmap_max=64,
            resolution_schedule=[4],  # single resolution matching data
            steps_per_phase=50,
            lr=1e-3,
        )

        trainer = pl.Trainer(
            max_steps=10,
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model, loader)
        assert model._step_count > 0

    def test_generates_correct_shape(self, tiny_zarr_store):
        """Trained GAN generates volumes of expected shape."""
        from nobrainer.models.generative.progressivegan import ProgressiveGAN

        model = ProgressiveGAN(
            latent_size=64,
            fmap_base=128,
            fmap_max=64,
            resolution_schedule=[4, 8, 16],
            steps_per_phase=5,
        )
        model.generator.current_level = 2
        model.generator.alpha = 1.0
        model.eval()

        with torch.no_grad():
            z = torch.randn(2, 64)
            out = model.generator(z)

        assert out.shape == (2, 1, 16, 16, 16)
        assert torch.isfinite(out).all()

    def test_level_mapping_integration(self, tiny_zarr_store):
        """Level mapping works with real store metadata."""
        sys.path.insert(
            0,
            str(Path(__file__).resolve().parents[2] / "scripts" / "progressive_gan"),
        )
        from train import map_pyramid_to_gan_stages

        from nobrainer.datasets.zarr_store import store_info

        info = store_info(tiny_zarr_store)
        n_levels = info.get("n_levels", 1)
        volume_shape = tuple(info["volume_shape"])

        mapping = map_pyramid_to_gan_stages(
            n_levels,
            resolution_schedule=[4, 8, 16],
            volume_shape=volume_shape,
        )

        assert len(mapping) == 3
        assert all(0 <= lvl < n_levels for lvl in mapping)
        # Finest stage → level 0 (full res)
        assert mapping[-1] == 0
        # Coarsest stage → highest valid level
        assert mapping[0] == n_levels - 1
