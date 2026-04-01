"""SegFormer3D: Efficient Transformer for 3D Medical Image Segmentation.

Port of SegFormer3D (Perera et al., CVPR 2024 Workshop) to nobrainer.
Hierarchical vision transformer with efficient self-attention and
all-MLP decoder for 3D volumetric segmentation.

Reference: https://arxiv.org/abs/2404.10156
Original: https://github.com/OSUPCVLab/SegFormer3D
"""

from __future__ import annotations

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Encoder components
# ---------------------------------------------------------------------------


class PatchEmbedding3d(nn.Module):
    """3D overlapping patch embedding via strided convolution.

    Parameters
    ----------
    in_channels : int
        Input channels.
    embed_dim : int
        Output embedding dimension.
    kernel_size : int
        Conv kernel size.
    stride : int
        Conv stride (< kernel_size for overlap).
    padding : int
        Conv padding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Returns (B, N, C) tensor and spatial dims (D, H, W)."""
        x = self.proj(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = self.norm(x)
        return x, D, H, W


class EfficientSelfAttention3d(nn.Module):
    """Multi-head self-attention with spatial reduction.

    Reduces K, V spatial dimensions by ``sr_ratio`` before attention,
    giving O(N²/R²) complexity instead of O(N²).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 1,
        sr_ratio: int = 8,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        D: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)

        if self.sr_ratio > 1:
            x_3d = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
            x_sr = self.sr(x_3d)
            x_sr = rearrange(x_sr, "b c d h w -> b (d h w) c")
            x_sr = self.sr_norm(x_sr)
            kv = self.kv(x_sr)
        else:
            kv = self.kv(x)

        kv = rearrange(kv, "b n (two h d) -> two b h n d", two=2, h=self.num_heads)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)


class DWConv3d(nn.Module):
    """3D depth-wise convolution for positional encoding in MLP."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        x = self.bn(self.dwconv(x))
        x = rearrange(x, "b c d h w -> b (d h w) c")
        return x


class MixFFN3d(nn.Module):
    """Feed-forward network with depth-wise conv for positional encoding."""

    def __init__(
        self, embed_dim: int = 64, mlp_ratio: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.dwconv = DWConv3d(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock3d(nn.Module):
    """Transformer block: LN → Attention → residual → LN → FFN → residual."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 1,
        mlp_ratio: int = 4,
        sr_ratio: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientSelfAttention3d(
            embed_dim,
            num_heads,
            sr_ratio,
            qkv_bias=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MixFFN3d(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), D, H, W)
        x = x + self.ffn(self.norm2(x), D, H, W)
        return x


# ---------------------------------------------------------------------------
# Hierarchical Encoder (Mix Transformer)
# ---------------------------------------------------------------------------


class MixTransformerEncoder3d(nn.Module):
    """4-stage hierarchical transformer encoder.

    Each stage: PatchEmbedding → N × TransformerBlock → output features.
    Spatial resolution halves (approximately) at each stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dims: tuple[int, ...] = (64, 128, 320, 512),
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (1, 2, 5, 8),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
        mlp_ratio: int = 4,
        patch_sizes: tuple[int, ...] = (7, 3, 3, 3),
        strides: tuple[int, ...] = (4, 2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_stages = len(embed_dims)

        for i in range(self.num_stages):
            in_ch = in_channels if i == 0 else embed_dims[i - 1]
            padding = patch_sizes[i] // 2
            patch_embed = PatchEmbedding3d(
                in_ch,
                embed_dims[i],
                patch_sizes[i],
                strides[i],
                padding,
            )
            blocks = nn.ModuleList(
                [
                    TransformerBlock3d(
                        embed_dims[i],
                        num_heads[i],
                        mlp_ratio,
                        sr_ratios[i],
                        dropout,
                    )
                    for _ in range(depths[i])
                ]
            )
            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed_{i}", patch_embed)
            setattr(self, f"blocks_{i}", blocks)
            setattr(self, f"norm_{i}", norm)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns list of multi-scale features [(B, C_i, D_i, H_i, W_i)]."""
        features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed_{i}")
            blocks = getattr(self, f"blocks_{i}")
            norm = getattr(self, f"norm_{i}")

            x, D, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, D, H, W)
            x = norm(x)

            # Reshape back to 3D for next stage
            x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
            features.append(x)

        return features


# ---------------------------------------------------------------------------
# MLP Decoder
# ---------------------------------------------------------------------------


class SegFormerDecoderHead(nn.Module):
    """All-MLP decoder that aggregates multi-scale features.

    Upsamples features from all encoder stages to the highest resolution,
    concatenates, and projects to n_classes.
    """

    def __init__(
        self,
        embed_dims: tuple[int, ...] = (64, 128, 320, 512),
        decoder_dim: int = 256,
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.n_stages = len(embed_dims)

        # Linear projection per stage
        self.linears = nn.ModuleList(
            [nn.Linear(embed_dims[i], decoder_dim) for i in range(self.n_stages)]
        )

        # Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Linear(decoder_dim * self.n_stages, decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Linear(decoder_dim, n_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """features: list of (B, C_i, D_i, H_i, W_i) from encoder stages."""
        # Target spatial size = largest feature map (first stage)
        target = features[0].shape[2:]  # (D0, H0, W0)

        projected = []
        for i, feat in enumerate(features):
            B, C, D, H, W = feat.shape
            x = rearrange(feat, "b c d h w -> b (d h w) c")
            x = self.linears[i](x)  # (B, N, decoder_dim)
            x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)

            # Upsample to target resolution
            if (D, H, W) != target:
                x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)

            projected.append(x)

        # Concatenate along channel dim, then fuse
        fused = torch.cat(projected, dim=1)  # (B, decoder_dim * n_stages, D, H, W)
        B, C, D, H, W = fused.shape
        fused = rearrange(fused, "b c d h w -> b (d h w) c")
        fused = self.fuse(fused)
        out = self.pred(fused)  # (B, D*H*W, n_classes)
        out = rearrange(out, "b (d h w) c -> b c d h w", d=D, h=H, w=W)

        return out


# ---------------------------------------------------------------------------
# SegFormer3D Model
# ---------------------------------------------------------------------------


class SegFormer3D(nn.Module):
    """SegFormer3D: Hierarchical Transformer for 3D Medical Image Segmentation.

    Combines a multi-stage transformer encoder (MixTransformer) with an
    all-MLP decoder for efficient 3D segmentation.

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input channels (1 for MRI).
    embed_dims : tuple of int
        Embedding dimensions per encoder stage.
    depths : tuple of int
        Number of transformer blocks per stage.
    num_heads : tuple of int
        Number of attention heads per stage.
    sr_ratios : tuple of int
        Spatial reduction ratios for efficient attention per stage.
    mlp_ratio : int
        MLP hidden dimension multiplier.
    decoder_dim : int
        Decoder unified channel dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        embed_dims: tuple[int, ...] = (32, 64, 160, 256),
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (1, 2, 5, 8),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
        mlp_ratio: int = 4,
        decoder_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MixTransformerEncoder3d(
            in_channels=in_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.decoder = SegFormerDecoderHead(
            embed_dims=embed_dims,
            decoder_dim=decoder_dim,
            n_classes=n_classes,
        )

        # Final upsample to match input resolution
        self._upsample_factor = 4  # first stage stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) → (B, n_classes, D, H, W)."""
        input_shape = x.shape[2:]
        features = self.encoder(x)
        out = self.decoder(features)

        # Upsample to input resolution if needed
        if out.shape[2:] != input_shape:
            out = F.interpolate(
                out, size=input_shape, mode="trilinear", align_corners=False
            )

        return out


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def segformer3d(
    n_classes: int = 1,
    in_channels: int = 1,
    embed_dims: tuple[int, ...] = (32, 64, 160, 256),
    depths: tuple[int, ...] = (2, 2, 2, 2),
    num_heads: tuple[int, ...] = (1, 2, 5, 8),
    sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
    mlp_ratio: int = 4,
    decoder_dim: int = 256,
    dropout: float = 0.0,
    **kwargs,
) -> SegFormer3D:
    """Factory function for :class:`SegFormer3D`.

    Default config (~4.5M params) matches the paper's base variant.

    Common size variants:
    - **tiny**: ``embed_dims=(16, 32, 80, 128)`` (~1.5M params)
    - **small** (default): ``embed_dims=(32, 64, 160, 256)`` (~4.5M params)
    - **base**: ``embed_dims=(64, 128, 320, 512)`` (~18M params)
    """
    return SegFormer3D(
        n_classes=n_classes,
        in_channels=in_channels,
        embed_dims=embed_dims,
        depths=depths,
        num_heads=num_heads,
        sr_ratios=sr_ratios,
        mlp_ratio=mlp_ratio,
        decoder_dim=decoder_dim,
        dropout=dropout,
    )


__all__ = ["SegFormer3D", "segformer3d"]
