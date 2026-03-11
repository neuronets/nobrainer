"""Custom padding layers for nobrainer (PyTorch)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroPadding3DChannels(nn.Module):
    """Pad the channel dimension of a 5-D tensor symmetrically with zeros.

    Expects input of shape ``(N, C, D, H, W)`` and pads ``C`` by
    ``padding`` on each side, yielding ``(N, C + 2*padding, D, H, W)``.

    Parameters
    ----------
    padding : int
        Number of zero channels to prepend and append.
    """

    def __init__(self, padding: int) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad pads in reverse dim order; last two entries pad dim 1 (C)
        return F.pad(x, (0, 0, 0, 0, 0, 0, self.padding, self.padding))

    def extra_repr(self) -> str:
        return f"padding={self.padding}"
