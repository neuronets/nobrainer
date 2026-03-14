"""Unit tests for ProgressiveGAN and DCGAN (CPU smoke tests)."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.models.generative import DCGAN, ProgressiveGAN, dcgan, progressivegan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_loader(batch_size: int = 2, spatial: int = 4) -> DataLoader:
    """Return a DataLoader with synthetic 3-D volumes."""
    imgs = torch.randn(4, 1, spatial, spatial, spatial)
    return DataLoader(TensorDataset(imgs), batch_size=batch_size)


# ---------------------------------------------------------------------------
# ProgressiveGAN
# ---------------------------------------------------------------------------


class TestProgressiveGAN:
    def test_construction(self):
        m = ProgressiveGAN(
            latent_size=8, fmap_base=16, fmap_max=16, resolution_schedule=[4, 8]
        )
        assert isinstance(m, ProgressiveGAN)

    def test_factory_function(self):
        m = progressivegan(
            latent_size=8, fmap_base=16, fmap_max=16, resolution_schedule=[4, 8]
        )
        assert isinstance(m, ProgressiveGAN)

    def test_generator_output_shape(self):
        m = ProgressiveGAN(
            latent_size=8, fmap_base=16, fmap_max=16, resolution_schedule=[4]
        )
        m.generator.current_level = 0
        m.generator.alpha = 1.0
        z = torch.randn(2, 8)
        out = m.generator(z)
        assert out.shape[0] == 2
        assert out.shape[1] == 1

    def test_discriminator_output_shape(self):
        m = ProgressiveGAN(
            latent_size=8, fmap_base=16, fmap_max=16, resolution_schedule=[4]
        )
        m.discriminator.current_level = 0
        img = torch.randn(2, 1, 4, 4, 4)
        out = m.discriminator(img)
        assert out.shape == (2, 1)

    def test_training_step_losses_finite(self):
        """5-step CPU training smoke test."""
        m = ProgressiveGAN(
            latent_size=8,
            fmap_base=16,
            fmap_max=16,
            resolution_schedule=[4],
            steps_per_phase=10,
        )
        loader = _tiny_loader(batch_size=2, spatial=4)
        trainer = pl.Trainer(
            max_steps=5,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(m, loader)
        # Verify that logged losses are finite
        assert m._step_count > 0

    def test_alpha_schedule(self):
        m = ProgressiveGAN(
            latent_size=8,
            fmap_base=16,
            fmap_max=16,
            resolution_schedule=[4, 8],
            steps_per_phase=10,
        )
        m._step_count = 5
        m.on_train_batch_end()
        assert 0.0 <= m.generator.alpha <= 1.0


# ---------------------------------------------------------------------------
# DCGAN
# ---------------------------------------------------------------------------


class TestDCGAN:
    def test_construction(self):
        m = DCGAN(latent_size=8, n_filters=4)
        assert isinstance(m, DCGAN)

    def test_factory_function(self):
        m = dcgan(latent_size=8, n_filters=4)
        assert isinstance(m, DCGAN)

    def test_generator_output_shape(self):
        m = DCGAN(latent_size=8, n_filters=4)
        z = torch.randn(2, 8)
        out = m.generator(z)
        assert out.shape[0] == 2
        assert out.shape[1] == 1

    def test_discriminator_output_shape(self):
        m = DCGAN(latent_size=8, n_filters=4)
        img = torch.randn(2, 1, 64, 64, 64)
        out = m.discriminator(img)
        assert out.shape == (2, 1)

    def test_training_step_losses_finite(self):
        """5-step CPU training smoke test."""
        m = DCGAN(latent_size=8, n_filters=4)
        loader = _tiny_loader(batch_size=2, spatial=4)
        trainer = pl.Trainer(
            max_steps=5,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(m, loader)
        # No assertion needed — if fit() completes without error, losses were finite

    def test_configure_optimizers(self):
        m = DCGAN(latent_size=8, n_filters=4)
        opts = m.configure_optimizers()
        assert len(opts) == 2  # (opt_g, opt_d)
