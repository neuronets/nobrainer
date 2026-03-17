"""Tests for brain generation with Progressive GAN."""

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pl = pytest.importorskip("pytorch_lightning")  # noqa: F841

from nobrainer.processing import Generation  # noqa: E402


class TestBrainGeneration:
    """Test generative model training and image generation."""

    def test_generate_returns_nifti_images(self, sample_data):
        """Generation.fit().generate(2) returns 2 NIfTI images."""
        from scipy.ndimage import zoom

        # Downsample real volumes to 4^3 (GAN needs small, uniform volumes)
        volumes = []
        for img_path, _ in sample_data[:4]:
            vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
            vmin, vmax = vol.min(), vol.max()
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            factors = [4 / s for s in vol.shape[:3]]
            volumes.append(zoom(vol, factors, order=1))

        imgs = torch.from_numpy(np.stack(volumes)[:, None])  # (N, 1, 4, 4, 4)
        loader = DataLoader(TensorDataset(imgs), batch_size=2, shuffle=True)

        gen = Generation(
            "progressivegan",
            model_args={
                "latent_size": 16,
                "fmap_base": 16,
                "fmap_max": 16,
                "resolution_schedule": [4],
                "steps_per_phase": 100,
            },
        )
        gen.fit(loader, epochs=50)
        images = gen.generate(n_images=2)

        assert len(images) == 2
        for img in images:
            assert isinstance(img, nib.Nifti1Image)
            assert len(img.shape) >= 3
