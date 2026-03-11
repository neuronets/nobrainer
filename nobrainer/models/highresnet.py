"""HighResNet 3-D segmentation model (PyTorch).

Reference
---------
Li W. et al., "On the Compactness, Efficiency, and Representation of
3D Convolutional Networks: Brain Parcellation as a Pretext Task",
IPMI 2017. arXiv:1707.01992.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """Residual block: BN→Act→Conv→BN→Act→Conv + skip."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        act: type[nn.Module],
    ) -> None:
        super().__init__()
        padding = dilation
        self.path = nn.Sequential(
            nn.BatchNorm3d(channels),
            act(),
            nn.Conv3d(
                channels, channels, 3, padding=padding, dilation=dilation, bias=False
            ),
            nn.BatchNorm3d(channels),
            act(),
            nn.Conv3d(
                channels, channels, 3, padding=padding, dilation=dilation, bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.path(x)


class _ZeroPadChannels(nn.Module):
    """Pad the channel dimension symmetrically with zeros."""

    def __init__(self, extra_channels: int) -> None:
        super().__init__()
        self.pad = extra_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, 0, 0, 0, 0, self.pad, self.pad))


class HighResNet(nn.Module):
    """HighResNet — three stages of residual blocks with increasing dilation.

    Stage 1 (dilation=1): base_filters channels, n_blocks residual blocks
    Stage 2 (dilation=2): 2*base_filters channels
    Stage 3 (dilation=4): 4*base_filters channels

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input image channels.
    base_filters : int
        Initial feature map count (doubled each stage).
    n_blocks : int
        Number of residual blocks per stage.
    activation : str
        ``"relu"`` or ``"elu"``.
    dropout_rate : float
        Spatial dropout probability after the last stage (0 = none).
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        base_filters: int = 16,
        n_blocks: int = 3,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        act_cls: type[nn.Module] = {"relu": nn.ReLU, "elu": nn.ELU}[activation.lower()]
        f = base_filters  # 16

        # Initial projection to base_filters channels
        self.init_conv = nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False)

        # Stage 1: f channels, dilation 1 → pad to 3f
        s1 = [_ResBlock(f, dilation=1, act=act_cls) for _ in range(n_blocks)]
        self.stage1 = nn.Sequential(*s1)
        self.pad1 = _ZeroPadChannels(f)  # f → 3f

        # Stage 2: project 3f → 2f, dilation 2 → pad to 6f
        self.stage2_proj = nn.Conv3d(3 * f, 2 * f, kernel_size=1, bias=False)
        s2 = [_ResBlock(2 * f, dilation=2, act=act_cls) for _ in range(n_blocks)]
        self.stage2 = nn.Sequential(*s2)
        self.pad2 = _ZeroPadChannels(2 * f)  # 2f → 6f

        # Stage 3: project 6f → 4f, dilation 4
        self.stage3_proj = nn.Conv3d(6 * f, 4 * f, kernel_size=1, bias=False)
        s3 = [_ResBlock(4 * f, dilation=4, act=act_cls) for _ in range(n_blocks)]
        self.stage3 = nn.Sequential(*s3)

        self.dropout = (
            nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm3d(4 * f),
            act_cls(),
            nn.Conv3d(4 * f, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        s1 = self.stage1(x)
        s1 = self.pad1(s1)  # (N, 3f, D, H, W)

        s2 = self.stage2_proj(s1)
        s2 = self.stage2(s2)
        s2 = self.pad2(s2)  # (N, 6f, D, H, W)

        s3 = self.stage3_proj(s2)
        s3 = self.stage3(s3)
        s3 = self.dropout(s3)

        return self.classifier(s3)


def highresnet(
    n_classes: int = 1,
    in_channels: int = 1,
    base_filters: int = 16,
    n_blocks: int = 3,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    **kwargs,
) -> HighResNet:
    """Factory function for :class:`HighResNet`."""
    return HighResNet(
        n_classes=n_classes,
        in_channels=in_channels,
        base_filters=base_filters,
        n_blocks=n_blocks,
        activation=activation,
        dropout_rate=dropout_rate,
    )


__all__ = ["HighResNet", "highresnet"]
