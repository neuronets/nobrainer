"""MeshNet 3-D segmentation model (PyTorch).

Reference
---------
Fedorov A. et al., "End-to-end learning of brain tissue segmentation
from imperfect labeling", IJCNN 2017. arXiv:1612.00940.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Dilation schedules indexed by receptive field size
_DILATION_SCHEDULES: dict[int, list[int]] = {
    37: [1, 1, 1, 2, 4, 8, 1],
    67: [1, 1, 2, 4, 8, 16, 1],
    129: [1, 2, 4, 8, 16, 32, 1],
}


class _ConvBNActDrop(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        act: type[nn.Module],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        padding = dilation  # same-padding for 3×3×3 kernel
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            act(),
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MeshNet(nn.Module):
    """3-D MeshNet segmentation network.

    Seven layers of dilated 3×3×3 convolutions with a learnable dilation
    schedule that controls the receptive field.

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input image channels (1 for single-modality MRI).
    filters : int
        Number of feature maps in all hidden layers.
    receptive_field : int
        One of ``37``, ``67``, ``129`` — selects the dilation schedule.
    activation : str
        ``"relu"`` or ``"elu"``.
    dropout_rate : float
        Spatial dropout probability applied after each conv layer (0 = none).
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        filters: int = 71,
        receptive_field: int = 67,
        activation: str = "relu",
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if receptive_field not in _DILATION_SCHEDULES:
            raise ValueError(
                f"receptive_field must be one of {list(_DILATION_SCHEDULES)}, "
                f"got {receptive_field}"
            )
        dilations = _DILATION_SCHEDULES[receptive_field]
        act_cls: type[nn.Module] = {"relu": nn.ReLU, "elu": nn.ELU}[activation.lower()]

        layers: list[nn.Module] = []
        for i, dil in enumerate(dilations):
            in_ch = in_channels if i == 0 else filters
            layers.append(_ConvBNActDrop(in_ch, filters, dil, act_cls, dropout_rate))

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Conv3d(filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def meshnet(
    n_classes: int = 1,
    in_channels: int = 1,
    filters: int = 71,
    receptive_field: int = 67,
    activation: str = "relu",
    dropout_rate: float = 0.25,
    **kwargs,
) -> MeshNet:
    """Factory function for :class:`MeshNet`."""
    return MeshNet(
        n_classes=n_classes,
        in_channels=in_channels,
        filters=filters,
        receptive_field=receptive_field,
        activation=activation,
        dropout_rate=dropout_rate,
    )


__all__ = ["MeshNet", "meshnet"]
