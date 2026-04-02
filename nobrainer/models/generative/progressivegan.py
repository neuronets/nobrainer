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
    def __init__(self, in_ch: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _FromRGB(nn.Module):
    def __init__(self, out_ch: int, in_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), 0.2)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class _Generator(nn.Module):
    """Progressive generator.  Each stage doubles the spatial resolution.

    Supports optional conditioning via a condition vector that is
    concatenated with the latent vector before the initial block.
    """

    def __init__(
        self,
        latent_size: int,
        fmap_base: int,
        fmap_max: int,
        resolution_schedule: list[int],
        condition_size: int = 0,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.resolution_schedule = resolution_schedule
        self.current_level = 0
        self.condition_size = condition_size

        effective_latent = latent_size + condition_size

        def nf(level: int) -> int:
            return min(int(fmap_base / (2**level)), fmap_max)

        # Level 0: (latent + condition) → 4³ feature map
        self.init_block = nn.Sequential(
            nn.ConvTranspose3d(
                effective_latent, nf(0), kernel_size=4, stride=1, padding=0
            ),
            nn.LeakyReLU(0.2),
            _ConvBlock(nf(0), nf(0)),
        )
        self.to_rgb_blocks = nn.ModuleList([_ToRGB(nf(0), out_channels=out_channels)])
        self.upsample_blocks = nn.ModuleList()

        for level in range(1, len(resolution_schedule)):
            block = nn.Sequential(
                _ConvBlock(nf(level - 1), nf(level)),
                _ConvBlock(nf(level), nf(level)),
            )
            self.upsample_blocks.append(block)
            self.to_rgb_blocks.append(_ToRGB(nf(level), out_channels=out_channels))

        self.alpha: float = 1.0

    def forward(
        self, z: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        z : Tensor
            Latent vector ``(B, latent_size)``.
        condition : Tensor or None
            Conditioning vector ``(B, condition_size)``.  Concatenated
            with ``z`` before the initial block.  Required if the model
            was created with ``condition_size > 0``.
        """
        if self.condition_size > 0:
            if condition is None:
                raise ValueError("condition required when condition_size > 0")
            z = torch.cat([z, condition], dim=1)

        x = self.init_block(z.view(*z.shape, 1, 1, 1))

        if self.current_level == 0:
            return torch.tanh(self.to_rgb_blocks[0](x))

        for i in range(self.current_level - 1):
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = self.upsample_blocks[i](x)

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
    """Progressive discriminator.  Mirror of the generator.

    Supports conditioning via projection: the condition vector is
    projected and added to the discriminator's output logit
    (Miyato & Koyama, 2018).
    """

    def __init__(
        self,
        fmap_base: int,
        fmap_max: int,
        resolution_schedule: list[int],
        condition_size: int = 0,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.resolution_schedule = resolution_schedule
        self.current_level = 0
        self.condition_size = condition_size

        def nf(level: int) -> int:
            return min(int(fmap_base / (2**level)), fmap_max)

        # Level 0 (4³): feature → 1 (real/fake)
        self.final_block = nn.Sequential(
            _ConvBlock(nf(0), nf(0), use_pixel_norm=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(nf(0), 1),
        )
        self.from_rgb_blocks = nn.ModuleList([_FromRGB(nf(0), in_channels=in_channels)])
        self.downsample_blocks = nn.ModuleList()

        for level in range(1, len(resolution_schedule)):
            block = nn.Sequential(
                _ConvBlock(nf(level), nf(level), use_pixel_norm=False),
                _ConvBlock(nf(level), nf(level - 1), use_pixel_norm=False),
            )
            self.downsample_blocks.append(block)
            self.from_rgb_blocks.append(_FromRGB(nf(level), in_channels=in_channels))

        # Projection head for conditioning (Miyato & Koyama, 2018)
        if condition_size > 0:
            self.cond_proj = nn.Linear(condition_size, nf(0), bias=False)
        else:
            self.cond_proj = None

        self.alpha: float = 1.0

    def forward(
        self,
        img: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        img : Tensor
            Input image ``(B, C, D, H, W)``.
        condition : Tensor or None
            Conditioning vector ``(B, condition_size)`` for projection
            discrimination.
        """
        if self.current_level == 0:
            x = self.from_rgb_blocks[0](img)
            return self._output(x, condition)

        prev_img = F.avg_pool3d(img, kernel_size=2, stride=2)
        prev_x = self.from_rgb_blocks[self.current_level - 1](prev_img)

        x = self.from_rgb_blocks[self.current_level](img)
        x = self.downsample_blocks[self.current_level - 1](x)
        x = F.avg_pool3d(x, kernel_size=2, stride=2)

        x = self.alpha * x + (1.0 - self.alpha) * prev_x

        for i in range(self.current_level - 2, -1, -1):
            x = self.downsample_blocks[i](x)
            x = F.avg_pool3d(x, kernel_size=2, stride=2)

        return self._output(x, condition)

    def _output(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        """Compute final logit with optional projection conditioning."""
        # Pass through final block up to (but not including) the linear
        feat = self.final_block[:-1](x)  # conv + pool + flatten → (B, nf0)
        logit = self.final_block[-1](feat)  # linear → (B, 1)

        if self.cond_proj is not None and condition is not None:
            # Projection discrimination: logit += <embed(c), feat>
            c_embed = self.cond_proj(condition)  # (B, nf0)
            logit = logit + (feat * c_embed).sum(dim=1, keepdim=True)

        return logit


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class ProgressiveGAN(pl.LightningModule):
    """ProgressiveGAN as a PyTorch Lightning module.

    Supports optional conditioning (e.g., modality labels, class IDs, or
    embedding vectors from a conditioning image) and data augmentation.

    Parameters
    ----------
    latent_size : int
        Dimension of the latent noise vector.
    condition_size : int
        Conditioning vector dimension (0 = unconditional).  For class
        conditioning, use a small integer (e.g., number of modalities).
        For image conditioning, use the output dim of an encoder.
    in_channels : int
        Number of input image channels (1 = single modality, >1 for
        multi-channel or concatenated modalities).
    out_channels : int
        Number of output image channels (matches ``in_channels`` for
        reconstruction-style generation).
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
    augment_fn : callable or None
        Differentiable augmentation function applied to real (and
        optionally fake) images before the discriminator.  Signature:
        ``augment_fn(images: Tensor) -> Tensor``.  If None, no
        augmentation is applied.
    augment_fake : bool
        If True and ``augment_fn`` is set, also augment fake images
        before the discriminator (recommended for ADA-style training).
    """

    def __init__(
        self,
        latent_size: int = 512,
        condition_size: int = 0,
        in_channels: int = 1,
        out_channels: int = 1,
        fmap_base: int = 2048,
        fmap_max: int = 512,
        resolution_schedule: list[int] | None = None,
        steps_per_phase: int = 1000,
        lambda_gp: float = 10.0,
        lr: float = 1e-3,
        augment_fn: Any | None = None,
        augment_fake: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augment_fn"])
        if resolution_schedule is None:
            resolution_schedule = [4, 8, 16, 32, 64]
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.resolution_schedule = resolution_schedule
        self.steps_per_phase = steps_per_phase
        self.lambda_gp = lambda_gp
        self.lr = lr
        self.augment_fn = augment_fn
        self.augment_fake = augment_fake

        self.generator = _Generator(
            latent_size,
            fmap_base,
            fmap_max,
            resolution_schedule,
            condition_size=condition_size,
            out_channels=out_channels,
        )
        self.discriminator = _Discriminator(
            fmap_base,
            fmap_max,
            resolution_schedule,
            condition_size=condition_size,
            in_channels=in_channels,
        )

        self._step_count = 0
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation if configured."""
        if self.augment_fn is not None:
            return self.augment_fn(x)
        return x

    def _gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute WGAN-GP gradient penalty."""
        b = real.size(0)
        eps = torch.rand(b, 1, 1, 1, 1, device=real.device)
        interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
        d_interp = self.discriminator(interp, condition=condition)
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

        # Extract conditioning if present
        condition = None
        if self.condition_size > 0:
            if isinstance(batch, dict) and "condition" in batch:
                condition = batch["condition"]
            else:
                raise ValueError("condition_size > 0 but batch has no 'condition' key")

        # --- Discriminator step ---
        opt_d.zero_grad()
        fake = self.generator(z, condition=condition).detach()
        real_aug = self._augment(real)
        fake_aug = self._augment(fake) if self.augment_fake else fake
        d_real = self.discriminator(real_aug, condition=condition)
        d_fake = self.discriminator(fake_aug, condition=condition)
        gp = self._gradient_penalty(
            real, fake.requires_grad_(True), condition=condition
        )
        d_loss = d_fake.mean() - d_real.mean() + self.lambda_gp * gp
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Generator step ---
        opt_g.zero_grad()
        fake = self.generator(z, condition=condition)
        fake_aug = self._augment(fake) if self.augment_fake else fake
        g_loss = -self.discriminator(fake_aug, condition=condition).mean()
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
    condition_size: int = 0,
    in_channels: int = 1,
    out_channels: int = 1,
    fmap_base: int = 2048,
    fmap_max: int = 512,
    resolution_schedule: list[int] | None = None,
    **kwargs,
) -> ProgressiveGAN:
    """Factory function for :class:`ProgressiveGAN`."""
    return ProgressiveGAN(
        latent_size=latent_size,
        condition_size=condition_size,
        in_channels=in_channels,
        out_channels=out_channels,
        fmap_base=fmap_base,
        fmap_max=fmap_max,
        resolution_schedule=resolution_schedule,
        **kwargs,
    )


__all__ = ["ProgressiveGAN", "progressivegan"]
