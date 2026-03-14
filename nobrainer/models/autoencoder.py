"""Symmetric 3-D autoencoder (PyTorch).

Encodes a 3-D volume into a flat latent vector and reconstructs it via
transposed convolutions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Symmetric 3-D convolutional autoencoder.

    Dynamically builds encoder depth from the spatial size of the input.

    Parameters
    ----------
    input_shape : tuple of int
        Volume shape ``(D, H, W)`` (spatial dims only).
    in_channels : int
        Number of input channels (1 for single-modality MRI).
    encoding_dim : int
        Size of the flat latent code.
    n_base_filters : int
        Base filter count; doubled each encoder level.
    batchnorm : bool
        Whether to apply Batch Normalisation in conv blocks.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (64, 64, 64),
        in_channels: int = 1,
        encoding_dim: int = 512,
        n_base_filters: int = 16,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        D = input_shape[0]
        n_levels = int(math.log2(D))

        # Build encoder
        enc_layers: list[nn.Module] = []
        ch_in = in_channels
        self._enc_channels: list[int] = []
        for i in range(n_levels):
            ch_out = min(n_base_filters * (2**i), encoding_dim)
            self._enc_channels.append(ch_out)
            block: list[nn.Module] = [
                nn.Conv3d(
                    ch_in,
                    ch_out,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not batchnorm,
                ),
            ]
            if batchnorm:
                block.append(nn.BatchNorm3d(ch_out))
            block.append(nn.ReLU(inplace=True))
            enc_layers.extend(block)
            ch_in = ch_out

        self.encoder_conv = nn.Sequential(*enc_layers)
        self.encoder_fc = nn.Linear(ch_in, encoding_dim)

        # Build decoder (mirror of encoder)
        dec_ch = list(reversed(self._enc_channels))
        self.decoder_fc = nn.Linear(encoding_dim, dec_ch[0])

        dec_layers: list[nn.Module] = []
        all_out = dec_ch[1:] + [in_channels]
        for i, ch_out in enumerate(all_out):
            ch_in_d = dec_ch[i]
            is_last = i == len(all_out) - 1
            act_d: nn.Module = (
                nn.Sigmoid() if is_last else nn.LeakyReLU(0.2, inplace=True)
            )
            use_bn = batchnorm and not is_last
            block_d: list[nn.Module] = [
                nn.ConvTranspose3d(
                    ch_in_d, ch_out, kernel_size=4, stride=2, padding=1, bias=not use_bn
                ),
            ]
            if use_bn:
                block_d.append(nn.BatchNorm3d(ch_out))
            block_d.append(act_d)
            dec_layers.extend(block_d)

        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)  # (N, C, 1, 1, 1)
        return self.encoder_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(z.size(0), -1, 1, 1, 1)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def autoencoder(
    input_shape: tuple[int, int, int] = (64, 64, 64),
    in_channels: int = 1,
    encoding_dim: int = 512,
    n_base_filters: int = 16,
    batchnorm: bool = True,
    **kwargs,
) -> Autoencoder:
    """Factory function for :class:`Autoencoder`."""
    return Autoencoder(
        input_shape=input_shape,
        in_channels=in_channels,
        encoding_dim=encoding_dim,
        n_base_filters=n_base_filters,
        batchnorm=batchnorm,
    )


__all__ = ["Autoencoder", "autoencoder"]
