"""Bayesian V-Net: encoder-decoder segmentation with weight uncertainty.

Replaces the standard ``nn.Conv3d`` convolutions with
:class:`~nobrainer.models.bayesian.layers.BayesianConv3d` (mean-field
variational inference via Pyro), preserving the residual encoder-decoder
architecture of V-Net.

Reference
---------
Milletari F. et al., "V-Net: Fully Convolutional Neural Networks for
Volumetric Medical Image Segmentation", 3DV 2016. arXiv:1606.04797.
"""

from __future__ import annotations

from pyro.nn import PyroModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BayesianConv3d


class _BayesResBlock(PyroModule):
    """Two-layer residual block with BayesianConv3d and a skip connection."""

    def __init__(self, channels: int, prior_type: str = "standard_normal") -> None:
        super().__init__()
        self.conv1 = BayesianConv3d(
            channels, channels, kernel_size=3, padding=1, prior_type=prior_type
        )
        self.conv2 = BayesianConv3d(
            channels, channels, kernel_size=3, padding=1, prior_type=prior_type
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.elu(h + x)


class _EncoderBlock(PyroModule):
    """One encoder level: project channels → residual block → max-pool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        prior_type: str = "standard_normal",
    ) -> None:
        super().__init__()
        self.proj = BayesianConv3d(in_ch, out_ch, kernel_size=1, prior_type=prior_type)
        self.bn_proj = nn.BatchNorm3d(out_ch)
        self.res = _BayesResBlock(out_ch, prior_type=prior_type)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.elu(self.bn_proj(self.proj(x)))
        h = self.res(h)
        return self.pool(h), h  # (down-sampled, skip)


class _DecoderBlock(PyroModule):
    """One decoder level: up-sample → concat skip → project → residual block."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        prior_type: str = "standard_normal",
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.proj = BayesianConv3d(
            out_ch + skip_ch, out_ch, kernel_size=1, prior_type=prior_type
        )
        self.bn_proj = nn.BatchNorm3d(out_ch)
        self.res = _BayesResBlock(out_ch, prior_type=prior_type)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        if h.shape != skip.shape:
            h = F.interpolate(
                h, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        h = torch.cat([h, skip], dim=1)
        h = F.elu(self.bn_proj(self.proj(h)))
        return self.res(h)


class BayesianVNet(PyroModule):
    """3-D V-Net with Bayesian convolutional layers.

    All ``nn.Conv3d`` layers in the encoder and decoder are replaced with
    :class:`BayesianConv3d`.  Upsampling transposed convolutions remain
    deterministic.

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input image channels.
    base_filters : int
        Feature-map count at the first encoder level (doubles each level).
    levels : int
        Number of encoder/decoder levels (default 4).
    prior_type : str
        ``"standard_normal"`` or ``"laplace"`` — forwarded to Bayesian layers.
    kl_weight : float
        Scalar applied to the summed KL divergence when computing the ELBO.
        Stored as an attribute; not used internally during forward.
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        base_filters: int = 16,
        levels: int = 4,
        prior_type: str = "standard_normal",
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self._levels = levels

        ch = [base_filters * (2**i) for i in range(levels)]

        # Input projection
        self.input_proj = BayesianConv3d(
            in_channels, ch[0], kernel_size=3, padding=1, prior_type=prior_type
        )
        self.input_bn = nn.BatchNorm3d(ch[0])

        # Encoder — registered as individually named attributes so Pyro can
        # assign unique site names (nn.ModuleList does not propagate names).
        # encoder_i: ch[i] → ch[i+1]; skip tensor has ch[i+1] channels.
        for i in range(levels - 1):
            enc = _EncoderBlock(ch[i], ch[i + 1], prior_type)
            setattr(self, f"encoder_{i}", enc)

        # Bottom residual block (no pooling)
        self.bottom_res = _BayesResBlock(ch[-1], prior_type=prior_type)

        # Decoder — decoder_i processes the stage closest to the bottom first.
        # decoder_i: in_ch = ch[L-1-i], skip_ch = ch[L-1-i], out_ch = ch[L-2-i]
        # (upsampled out_ch channels are cat'd with skip_ch to give
        #  out_ch + skip_ch channels before the projection layer)
        L = levels
        for i in range(L - 1):
            in_ch = ch[L - 1 - i]
            skip_ch = ch[L - 1 - i]  # skip from encoder_{L-2-i} has ch[L-1-i] chans
            out_ch = ch[L - 2 - i]
            dec = _DecoderBlock(in_ch, skip_ch, out_ch, prior_type)
            setattr(self, f"decoder_{i}", dec)

        # Final 1×1×1 classifier — deterministic
        self.classifier = nn.Conv3d(ch[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.input_bn(self.input_proj(x)))

        skips: list[torch.Tensor] = []
        for i in range(self._levels - 1):
            enc = getattr(self, f"encoder_{i}")
            h, skip = enc(h)
            skips.append(skip)

        h = self.bottom_res(h)

        for i in range(self._levels - 1):
            dec = getattr(self, f"decoder_{i}")
            skip = skips[self._levels - 2 - i]
            h = dec(h, skip)

        return self.classifier(h)


def bayesian_vnet(
    n_classes: int = 1,
    in_channels: int = 1,
    base_filters: int = 16,
    levels: int = 4,
    prior_type: str = "standard_normal",
    kl_weight: float = 1.0,
    **kwargs,
) -> BayesianVNet:
    """Factory function for :class:`BayesianVNet`."""
    return BayesianVNet(
        n_classes=n_classes,
        in_channels=in_channels,
        base_filters=base_filters,
        levels=levels,
        prior_type=prior_type,
        kl_weight=kl_weight,
    )


__all__ = ["BayesianVNet", "bayesian_vnet"]
