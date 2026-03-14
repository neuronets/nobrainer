"""DCGAN implemented as a PyTorch Lightning module.

Standard alternating generator/discriminator training using BCE loss.

Reference
---------
Radford A. et al., "Unsupervised Representation Learning with Deep
Convolutional Generative Adversarial Networks", ICLR 2016. arXiv:1511.06434.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _GenBlock(nn.Module):
    """Transposed-conv + BN + ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DiscBlock(nn.Module):
    """Conv + (optional BN) + LeakyReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
        ]
        if use_bn:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class _DCGenerator(nn.Module):
    """4-level transposed-conv generator; outputs (N, 1, 32, 32, 32)."""

    def __init__(self, latent_size: int = 128, n_filters: int = 64) -> None:
        super().__init__()
        nf = n_filters
        self.net = nn.Sequential(
            # latent (N, Z, 1, 1, 1) → (N, nf*8, 4, 4, 4)
            nn.ConvTranspose3d(latent_size, nf * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(nf * 8),
            nn.ReLU(inplace=True),
            # (N, nf*8, 4, 4, 4) → (N, nf*4, 8, 8, 8)
            _GenBlock(nf * 8, nf * 4),
            # (N, nf*4, 8, 8, 8) → (N, nf*2, 16, 16, 16)
            _GenBlock(nf * 4, nf * 2),
            # (N, nf*2, 16, 16, 16) → (N, nf, 32, 32, 32)
            _GenBlock(nf * 2, nf),
            # (N, nf, 32, 32, 32) → (N, 1, 32, 32, 32)
            nn.ConvTranspose3d(nf, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(*z.shape, 1, 1, 1))


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class _DCDiscriminator(nn.Module):
    """4-level conv discriminator; expects (N, 1, 64, 64, 64)."""

    def __init__(self, n_filters: int = 64) -> None:
        super().__init__()
        nf = n_filters
        self.net = nn.Sequential(
            # (N, 1, 64, 64, 64) → (N, nf, 32, 32, 32); no BN on first layer
            _DiscBlock(1, nf, use_bn=False),
            # → (N, nf*2, 16, 16, 16)
            _DiscBlock(nf, nf * 2),
            # → (N, nf*4, 8, 8, 8)
            _DiscBlock(nf * 2, nf * 4),
            # → (N, nf*8, 4, 4, 4)
            _DiscBlock(nf * 4, nf * 8),
            # → (N, 1, 1, 1, 1)
            nn.Conv3d(nf * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class DCGAN(pl.LightningModule):
    """DCGAN as a PyTorch Lightning module.

    Uses binary cross-entropy (non-saturating G loss) with standard
    alternating G/D updates.

    Parameters
    ----------
    latent_size : int
        Dimension of the latent noise vector.
    n_filters : int
        Base channel count for generator and discriminator.
    lr : float
        Learning rate for Adam.
    beta1 : float
        Adam beta1.
    """

    def __init__(
        self,
        latent_size: int = 128,
        n_filters: int = 64,
        lr: float = 2e-4,
        beta1: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.latent_size = latent_size
        self.lr = lr
        self.beta1 = beta1

        self.generator = _DCGenerator(latent_size, n_filters)
        self.discriminator = _DCDiscriminator(n_filters)
        self.automatic_optimization = False

        # Fixed noise for visualisation
        self._fixed_z = None

    def _sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.latent_size, device=self.device)

    def training_step(self, batch: Any, batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()

        real = batch["image"] if isinstance(batch, dict) else batch[0]
        b = real.size(0)

        real_label = torch.ones(b, 1, device=self.device)
        fake_label = torch.zeros(b, 1, device=self.device)

        # --- Discriminator step ---
        opt_d.zero_grad()
        z = self._sample_z(b)
        fake = self.generator(z).detach()
        # Resize real to discriminator input size if necessary
        if real.shape[-1] != 64:
            real_in = F.interpolate(
                real, size=(64, 64, 64), mode="trilinear", align_corners=False
            )
        else:
            real_in = real
        if fake.shape[-1] != 64:
            fake_in = F.interpolate(
                fake, size=(64, 64, 64), mode="trilinear", align_corners=False
            )
        else:
            fake_in = fake
        d_real = F.binary_cross_entropy_with_logits(
            self.discriminator(real_in), real_label
        )
        d_fake = F.binary_cross_entropy_with_logits(
            self.discriminator(fake_in), fake_label
        )
        d_loss = (d_real + d_fake) * 0.5
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Generator step ---
        opt_g.zero_grad()
        z = self._sample_z(b)
        fake = self.generator(z)
        if fake.shape[-1] != 64:
            fake_in = F.interpolate(
                fake, size=(64, 64, 64), mode="trilinear", align_corners=False
            )
        else:
            fake_in = fake
        g_loss = F.binary_cross_entropy_with_logits(
            self.discriminator(fake_in), real_label
        )
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        return [opt_g, opt_d]


def dcgan(
    latent_size: int = 128,
    n_filters: int = 64,
    **kwargs,
) -> DCGAN:
    """Factory function for :class:`DCGAN`."""
    return DCGAN(latent_size=latent_size, n_filters=n_filters, **kwargs)


__all__ = ["DCGAN", "dcgan"]
