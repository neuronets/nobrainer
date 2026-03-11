"""MaxPool4D layer for PyTorch.

Implements 4-D max-pooling (N, C, V, D, H, W) by treating the volume
dimension V as a batch dimension and applying ``nn.MaxPool3d`` over
(D, H, W).  This avoids the need for a custom CUDA kernel.
"""

import torch
import torch.nn as nn


class MaxPool4D(nn.Module):
    """Max-pooling over 4 spatial dimensions.

    Expects input of shape ``(N, C, V, D, H, W)`` and applies
    ``kernel_size`` / ``stride`` / ``padding`` along the last 3
    dimensions (D, H, W).  The volume dimension V is reduced with
    ``pool_v`` if ``> 1``.

    Parameters
    ----------
    kernel_size : int or tuple
        Kernel size for the (D, H, W) axes.
    stride : int or tuple or None
        Stride; defaults to ``kernel_size``.
    padding : int or tuple
        Zero-padding added to all spatial sides.
    pool_v : int
        Max-pool kernel size along the volume (V) axis.  ``1`` leaves
        V unchanged.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
        pool_v: int = 1,
    ) -> None:
        super().__init__()
        self.pool3d = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
        self.pool_v = pool_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(
                f"MaxPool4D expects 6-D input (N, C, V, D, H, W), got {x.dim()}-D"
            )
        N, C, V, D, H, W = x.shape
        # Merge batch and volume dims so MaxPool3d sees (N*V, C, D, H, W)
        out = self.pool3d(x.view(N * V, C, D, H, W))
        _, _, D2, H2, W2 = out.shape
        out = out.view(N, C, V, D2, H2, W2)
        if self.pool_v > 1:
            # Max over V with stride pool_v (using unfold for non-overlapping)
            out = out.unfold(2, self.pool_v, self.pool_v).amax(dim=-1)
        return out

    def extra_repr(self) -> str:
        return f"pool3d={self.pool3d}, pool_v={self.pool_v}"
