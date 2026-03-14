"""GPU end-to-end test: ProgressiveGAN training.

T054 — US3 acceptance scenario: ProgressiveGAN completes extended training
on synthetic 3D volumes without NaN in losses. Generated output has correct
shape and non-trivial intensity distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.models.generative import ProgressiveGAN


def _make_loader(n_samples=64, spatial=4, batch_size=4):
    """Create a DataLoader with enough data for extended training."""
    imgs = torch.randn(n_samples, 1, spatial, spatial, spatial)
    return DataLoader(TensorDataset(imgs), batch_size=batch_size, shuffle=True)


@pytest.mark.gpu
class TestProgressiveGANEndToEnd:
    def test_extended_training_no_nan(self):
        """Train ProgressiveGAN for many steps; verify no NaN in discriminator."""
        torch.manual_seed(42)

        loader = _make_loader(n_samples=64, spatial=4, batch_size=4)

        model = ProgressiveGAN(
            latent_size=32,
            fmap_base=32,
            fmap_max=32,
            resolution_schedule=[4],
            steps_per_phase=2000,
        )

        trainer = pl.Trainer(
            max_steps=500,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)

        # Verify discriminator outputs are finite after training
        model.eval()
        with torch.no_grad():
            x_real = next(iter(loader))[0].to(model.device)
            z = torch.randn(x_real.size(0), 32, device=model.device)
            x_fake = model.generator(z)
            d_real = model.discriminator(x_real)
            d_fake = model.discriminator(x_fake)

        assert torch.isfinite(d_real).all(), "d_real contains NaN/Inf"
        assert torch.isfinite(d_fake).all(), "d_fake contains NaN/Inf"
        assert not torch.isnan(x_fake).any(), "Generated volumes contain NaN"

    def test_generated_output_shape(self):
        """After training, generated volumes have correct shape."""
        torch.manual_seed(42)

        loader = _make_loader(n_samples=32, spatial=4, batch_size=4)

        model = ProgressiveGAN(
            latent_size=32,
            fmap_base=32,
            fmap_max=32,
            resolution_schedule=[4],
            steps_per_phase=500,
        )

        trainer = pl.Trainer(
            max_steps=100,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)

        model.eval()
        model.generator.current_level = 0
        model.generator.alpha = 1.0
        with torch.no_grad():
            z = torch.randn(4, 32, device=model.device)
            generated = model.generator(z)

        # Check shape: (4, 1, 4, 4, 4)
        assert generated.shape == (
            4,
            1,
            4,
            4,
            4,
        ), f"Expected (4, 1, 4, 4, 4), got {generated.shape}"
        assert not np.isnan(generated.cpu().numpy()).any(), "NaN in generated"
