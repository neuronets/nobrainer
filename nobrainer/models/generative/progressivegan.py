"""ProgressiveGAN implemented as a PyTorch Lightning module.

Grows the generator and discriminator from 4³ to the target resolution in
stages.  Each stage fades in a new layer using a learnable ``alpha``
parameter that rises from 0 to 1 during the fade-in phase.

Reference
---------
Karras T. et al., "Progressive Growing of GANs for Improved Quality,
Stability, and Variation", ICLR 2018. arXiv:1710.10196.
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


def _pixel_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pixel-wise feature vector normalisation (ProGAN style)."""
    return x / (x.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_pixel_norm: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.use_pixel_norm = use_pixel_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv(x), 0.2)
        if self.use_pixel_norm:
            x = _pixel_norm(x)
        return x


class _ToRGB(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _FromRGB(nn.Module):
    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), 0.2)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class _Generator(nn.Module):
    """Progressive generator.  Each stage doubles the spatial resolution."""

    def __init__(
        self,
        latent_size: int,
        fmap_base: int,
        fmap_max: int,
        resolution_schedule: list[int],
    ) -> None:
        super().__init__()
        self.resolution_schedule = resolution_schedule
        self.current_level = 0

        def nf(level: int) -> int:
            return min(int(fmap_base / (2**level)), fmap_max)

        # Level 0: latent → 4³ feature map
        self.init_block = nn.Sequential(
            nn.ConvTranspose3d(latent_size, nf(0), kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            _ConvBlock(nf(0), nf(0)),
        )
        self.to_rgb_blocks = nn.ModuleList([_ToRGB(nf(0))])
        self.upsample_blocks = nn.ModuleList()

        for level in range(1, len(resolution_schedule)):
            block = nn.Sequential(
                _ConvBlock(nf(level - 1), nf(level)),
                _ConvBlock(nf(level), nf(level)),
            )
            self.upsample_blocks.append(block)
            self.to_rgb_blocks.append(_ToRGB(nf(level)))

        self.alpha: float = 1.0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.init_block(z.view(*z.shape, 1, 1, 1))

        if self.current_level == 0:
            return torch.tanh(self.to_rgb_blocks[0](x))

        # Grow through levels up to current_level - 1, then fade in last level
        for i in range(self.current_level - 1):
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = self.upsample_blocks[i](x)

        # Fade-in: blend previous RGB with new upsampled RGB
        prev_rgb = self.to_rgb_blocks[self.current_level - 1](x)
        prev_rgb = F.interpolate(
            prev_rgb, scale_factor=2, mode="trilinear", align_corners=False
        )

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.upsample_blocks[self.current_level - 1](x)
        new_rgb = self.to_rgb_blocks[self.current_level](x)

        out = self.alpha * new_rgb + (1.0 - self.alpha) * prev_rgb
        return torch.tanh(out)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class _Discriminator(nn.Module):
    """Progressive discriminator.  Mirror of the generator."""

    def __init__(
        self,
        fmap_base: int,
        fmap_max: int,
        resolution_schedule: list[int],
    ) -> None:
        super().__init__()
        self.resolution_schedule = resolution_schedule
        self.current_level = 0

        def nf(level: int) -> int:
            return min(int(fmap_base / (2**level)), fmap_max)

        # Level 0 (4³): feature → 1 (real/fake)
        self.final_block = nn.Sequential(
            _ConvBlock(nf(0), nf(0), use_pixel_norm=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(nf(0), 1),
        )
        self.from_rgb_blocks = nn.ModuleList([_FromRGB(nf(0))])
        self.downsample_blocks = nn.ModuleList()

        for level in range(1, len(resolution_schedule)):
            block = nn.Sequential(
                _ConvBlock(nf(level), nf(level), use_pixel_norm=False),
                _ConvBlock(nf(level), nf(level - 1), use_pixel_norm=False),
            )
            self.downsample_blocks.append(block)
            self.from_rgb_blocks.append(_FromRGB(nf(level)))

        self.alpha: float = 1.0

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.current_level == 0:
            x = self.from_rgb_blocks[0](img)
            return self.final_block(x)

        # Fade-in: blend downsampled previous level with new level
        prev_img = F.avg_pool3d(img, kernel_size=2, stride=2)
        prev_x = self.from_rgb_blocks[self.current_level - 1](prev_img)

        x = self.from_rgb_blocks[self.current_level](img)
        x = self.downsample_blocks[self.current_level - 1](x)
        x = F.avg_pool3d(x, kernel_size=2, stride=2)

        x = self.alpha * x + (1.0 - self.alpha) * prev_x

        for i in range(self.current_level - 2, -1, -1):
            x = self.downsample_blocks[i](x)
            x = F.avg_pool3d(x, kernel_size=2, stride=2)

        return self.final_block(x)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class ProgressiveGAN(pl.LightningModule):
    """ProgressiveGAN as a PyTorch Lightning module.

    Parameters
    ----------
    latent_size : int
        Dimension of the latent noise vector.
    label_size : int
        Conditioning label dimension (0 = unconditional).
    fmap_base : int
        Base feature-map count used to compute per-level channels.
    fmap_max : int
        Maximum feature-map count at any level.
    resolution_schedule : list[int]
        Spatial resolutions to train (e.g. ``[4, 8, 16, 32]``).
    steps_per_phase : int
        Number of training steps in each fade-in phase.
    lambda_gp : float
        WGAN-GP gradient penalty weight.
    lr : float
        Learning rate for Adam (used for both G and D).
    """

    def __init__(
        self,
        latent_size: int = 512,
        label_size: int = 0,
        fmap_base: int = 2048,
        fmap_max: int = 512,
        resolution_schedule: list[int] | None = None,
        steps_per_phase: int = 1000,
        lambda_gp: float = 10.0,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if resolution_schedule is None:
            resolution_schedule = [4, 8, 16, 32, 64]
        self.latent_size = latent_size
        self.resolution_schedule = resolution_schedule
        self.steps_per_phase = steps_per_phase
        self.lambda_gp = lambda_gp
        self.lr = lr

        self.generator = _Generator(
            latent_size, fmap_base, fmap_max, resolution_schedule
        )
        self.discriminator = _Discriminator(fmap_base, fmap_max, resolution_schedule)

        self._step_count = 0
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute WGAN-GP gradient penalty."""
        b = real.size(0)
        eps = torch.rand(b, 1, 1, 1, 1, device=real.device)
        interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
        d_interp = self.discriminator(interp)
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=[1, 2, 3, 4]) - 1) ** 2).mean()
        return gp

    def _sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.latent_size, device=self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()

        real = batch["image"] if isinstance(batch, dict) else batch[0]
        b = real.size(0)
        z = self._sample_z(b)

        # --- Discriminator step ---
        opt_d.zero_grad()
        fake = self.generator(z).detach()
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)
        gp = self._gradient_penalty(real, fake.requires_grad_(True))
        d_loss = d_fake.mean() - d_real.mean() + self.lambda_gp * gp
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Generator step ---
        opt_g.zero_grad()
        fake = self.generator(z)
        g_loss = -self.discriminator(fake).mean()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)
        self._step_count += 1

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        """Update alpha for fade-in scheduling."""
        n_levels = len(self.resolution_schedule)
        level = min(self._step_count // self.steps_per_phase, n_levels - 1)
        phase_step = self._step_count % self.steps_per_phase
        alpha = min(phase_step / max(self.steps_per_phase, 1), 1.0)
        self.generator.current_level = level
        self.discriminator.current_level = level
        self.generator.alpha = alpha
        self.discriminator.alpha = alpha

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.0, 0.99)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.0, 0.99)
        )
        return [opt_g, opt_d]


def progressivegan(
    latent_size: int = 512,
    label_size: int = 0,
    fmap_base: int = 2048,
    fmap_max: int = 512,
    resolution_schedule: list[int] | None = None,
    **kwargs,
) -> ProgressiveGAN:
    """Factory function for :class:`ProgressiveGAN`."""
    return ProgressiveGAN(
        latent_size=latent_size,
        label_size=label_size,
        fmap_base=fmap_base,
        fmap_max=fmap_max,
        resolution_schedule=resolution_schedule,
        **kwargs,
    )


__all__ = ["ProgressiveGAN", "progressivegan"]
