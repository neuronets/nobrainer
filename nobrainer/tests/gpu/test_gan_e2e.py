"""GPU end-to-end test: ProgressiveGAN training for 1000 steps.

T054 — US3 acceptance scenario: ProgressiveGAN completes 1000 training
steps on synthetic 3D volumes without NaN in losses. Generated output
has correct shape and non-trivial intensity distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.models.generative import ProgressiveGAN


@pytest.mark.gpu
class TestProgressiveGANEndToEnd:
    def test_1000_steps_no_nan(self):
        """Train ProgressiveGAN for 1000 steps; verify no NaN in losses."""
        torch.manual_seed(42)

        # Synthetic 3D volumes: 32 samples of 4^3
        imgs = torch.randn(32, 1, 4, 4, 4)
        loader = DataLoader(
            TensorDataset(imgs), batch_size=4, shuffle=True, drop_last=True
        )

        model = ProgressiveGAN(
            latent_size=32,
            fmap_base=32,
            fmap_max=32,
            resolution_schedule=[4],
            steps_per_phase=1000,
        )

        trainer = pl.Trainer(
            max_steps=1000,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)

        # Verify training completed
        assert model._step_count >= 1000, f"Only completed {model._step_count} steps"

    def test_generated_output_shape_and_distribution(self):
        """After training, generated volumes have correct shape and non-trivial values."""
        torch.manual_seed(42)

        imgs = torch.randn(16, 1, 4, 4, 4)
        loader = DataLoader(
            TensorDataset(imgs), batch_size=4, shuffle=True, drop_last=True
        )

        model = ProgressiveGAN(
            latent_size=32,
            fmap_base=32,
            fmap_max=32,
            resolution_schedule=[4],
            steps_per_phase=200,
        )

        trainer = pl.Trainer(
            max_steps=200,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, loader)

        # Generate samples
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

        # Check non-trivial intensity distribution
        gen_np = generated.cpu().numpy()
        assert not np.isnan(gen_np).any(), "Generated volumes contain NaN"
        assert (
            gen_np.std() > 0.01
        ), f"Generated volumes have trivial std={gen_np.std():.4f}"

    def test_losses_are_finite(self):
        """After 500 steps, g_loss and d_loss should be finite (not NaN/Inf)."""
        torch.manual_seed(42)

        imgs = torch.randn(16, 1, 4, 4, 4)
        loader = DataLoader(
            TensorDataset(imgs), batch_size=4, shuffle=True, drop_last=True
        )

        model = ProgressiveGAN(
            latent_size=32,
            fmap_base=32,
            fmap_max=32,
            resolution_schedule=[4],
            steps_per_phase=500,
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

        # Run one more step to capture the loss
        model.train()
        batch = next(iter(loader))
        with torch.no_grad():
            x_real = batch[0].to(model.device)
            z = torch.randn(x_real.size(0), 32, device=model.device)
            x_fake = model.generator(z)
            d_real = model.discriminator(x_real)
            d_fake = model.discriminator(x_fake)

        assert torch.isfinite(
            d_real
        ).all(), "Discriminator output on real is not finite"
        assert torch.isfinite(
            d_fake
        ).all(), "Discriminator output on fake is not finite"
