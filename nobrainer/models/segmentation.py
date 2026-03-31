"""MONAI-backed segmentation model factory functions.

All models expect input of shape ``(N, C_in, D, H, W)`` and produce
output of shape ``(N, n_classes, D, H, W)``.
"""

from __future__ import annotations

from monai.networks.nets import UNETR, AttentionUnet, UNet, VNet
import torch.nn as nn


def unet(
    n_classes: int = 1,
    in_channels: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 0,
    act: str = "RELU",
    norm: str = "BATCH",
    dropout: float = 0.0,
    **kwargs,
) -> UNet:
    """Return a 3-D UNet (MONAI implementation).

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input image channels (1 for grayscale MRI).
    channels : tuple of int
        Filter count at each level (len == levels + 1).
    strides : tuple of int
        Down-sampling stride at each level (len == levels).
    num_res_units : int
        Number of residual units per level (0 = plain conv blocks).
    act : str
        Activation name (MONAI convention: "RELU", "LEAKYRELU", "ELU", …).
    norm : str
        Normalisation: "BATCH", "INSTANCE", "GROUP", "LAYER", or "NONE".
    dropout : float
        Dropout probability (0 = disabled).
    """
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        act=act,
        norm=norm,
        dropout=dropout,
        **kwargs,
    )


def vnet(
    n_classes: int = 1,
    in_channels: int = 1,
    act: str = "elu",
    dropout_dim: int = 3,
    **kwargs,
) -> VNet:
    """Return a 3-D V-Net (MONAI implementation).

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input channels.
    act : str
        Activation function name (lowercase MONAI style: "elu", "relu", …).
    dropout_dim : int
        Dimension for spatial dropout (1 = channel, 3 = 3-D spatial).
    """
    return VNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        act=act,
        dropout_dim=dropout_dim,
        **kwargs,
    )


def attention_unet(
    n_classes: int = 1,
    in_channels: int = 1,
    channels: tuple[int, ...] = (64, 128, 256, 512),
    strides: tuple[int, ...] = (2, 2, 2),
    dropout: float = 0.0,
    **kwargs,
) -> AttentionUnet:
    """Return a 3-D Attention U-Net (MONAI implementation).

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input channels.
    channels : tuple of int
        Filter counts at each encoder level.
    strides : tuple of int
        Down-sampling strides (len == len(channels) - 1).
    dropout : float
        Dropout probability.
    """
    return AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        channels=channels,
        strides=strides,
        dropout=dropout,
        **kwargs,
    )


def unetr(
    n_classes: int = 1,
    in_channels: int = 1,
    img_size: tuple[int, int, int] = (96, 96, 96),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    dropout_rate: float = 0.1,
    norm_name: str = "instance",
    **kwargs,
) -> UNETR:
    """Return a UNETR (ViT backbone + U-Net decoder) (MONAI implementation).

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input channels.
    img_size : tuple of int
        Spatial size of the input volume ``(D, H, W)``.
    feature_size : int
        Spatial feature size for the decoder (MONAI default 16).
    hidden_size : int
        ViT embedding dimension (default 768 = ViT-B).
    mlp_dim : int
        MLP hidden dim in transformer blocks.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout applied inside the transformer.
    norm_name : str
        Normalisation: "instance", "batch".
    """
    return UNETR(
        in_channels=in_channels,
        out_channels=n_classes,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        norm_name=norm_name,
        **kwargs,
    )


def swin_unetr(
    n_classes: int = 1,
    in_channels: int = 1,
    feature_size: int = 24,
    depths: tuple[int, ...] = (2, 2, 2, 2),
    num_heads: tuple[int, ...] = (3, 6, 12, 24),
    norm_name: str = "instance",
    dropout_rate: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Return a SwinUNETR (Swin Transformer U-Net) (MONAI implementation).

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input channels.
    feature_size : int
        Feature size for the decoder (default 24).
    depths : tuple of int
        Number of Swin Transformer blocks at each stage.
    num_heads : tuple of int
        Number of attention heads at each stage.
    norm_name : str
        Normalisation: ``"instance"`` or ``"batch"``.
    dropout_rate : float
        Dropout probability.
    """
    from monai.networks.nets import SwinUNETR as _SwinUNETR

    return _SwinUNETR(
        in_channels=in_channels,
        out_channels=n_classes,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        norm_name=norm_name,
        drop_rate=dropout_rate,
        spatial_dims=3,
        **kwargs,
    )


def segresnet(
    n_classes: int = 1,
    in_channels: int = 1,
    blocks_down: tuple[int, ...] = (1, 2, 2, 4),
    init_filters: int = 16,
    norm: str = "INSTANCE",
    dropout_prob: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Return a SegResNet (residual encoder segmentation network) (MONAI).

    Used as the default architecture in MONAI Auto3DSeg.

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input channels.
    blocks_down : tuple of int
        Number of residual blocks at each encoder level.
    init_filters : int
        Initial number of filters (doubled at each level).
    norm : str
        Normalisation: ``"GROUP"``, ``"BATCH"``, ``"INSTANCE"``.
    dropout_prob : float
        Dropout probability.
    """
    from monai.networks.nets import SegResNet as _SegResNet

    return _SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_classes,
        blocks_down=blocks_down,
        init_filters=init_filters,
        norm=norm,
        dropout_prob=dropout_prob,
        **kwargs,
    )


__all__ = [
    "unet",
    "vnet",
    "attention_unet",
    "unetr",
    "swin_unetr",
    "segresnet",
]
