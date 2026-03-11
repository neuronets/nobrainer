"""SimSiam self-supervised learning model for 3-D brain volumes (PyTorch).

Reference
---------
Chen X. & He K., "Exploring Simple Siamese Representation Learning",
CVPR 2021. arXiv:2011.10566.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .highresnet import HighResNet


class SimSiam(nn.Module):
    """Siamese network with stop-gradient for self-supervised pre-training.

    Architecture
    ------------
    - **Backbone**: :class:`~nobrainer.models.highresnet.HighResNet` that
      encodes a 3-D volume into a spatial feature map.
    - **Projector**: Global average pool → MLP (2 hidden layers) → projection
      vector of size ``projection_dim``.
    - **Predictor**: Bottleneck MLP (``projection_dim`` → ``latent_dim`` →
      ``projection_dim``).

    Training
    --------
    Produce two augmented views of the same volume, pass each through the
    encoder + projector, and apply the *negative cosine similarity* loss
    between ``predictor(z1)`` and ``stop_grad(z2)`` (and vice-versa).

    Parameters
    ----------
    n_classes : int
        Passed to the HighResNet backbone (not used for classification,
        but kept for architecture compatibility).
    in_channels : int
        Number of input channels.
    projection_dim : int
        Output dimension of the projector head.
    latent_dim : int
        Hidden bottleneck size in the predictor.
    weight_decay : float
        L2 regularisation weight (applied externally via the optimiser).
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        projection_dim: int = 2048,
        latent_dim: int = 512,
        weight_decay: float = 0.0005,
    ) -> None:
        super().__init__()
        self.weight_decay = weight_decay

        backbone = HighResNet(n_classes=n_classes, in_channels=in_channels)
        # Determine backbone output channels by inspecting the classifier head
        self.backbone = backbone
        backbone_feat_ch = backbone.classifier[2].in_channels  # 4*f

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(backbone_feat_ch, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, projection_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone up to stage3 (before classifier head)."""
        h = self.backbone.init_conv(x)
        s1 = self.backbone.stage1(h)
        s1 = self.backbone.pad1(s1)
        s2 = self.backbone.stage2_proj(s1)
        s2 = self.backbone.stage2(s2)
        s2 = self.backbone.pad2(s2)
        s3 = self.backbone.stage3_proj(s2)
        s3 = self.backbone.stage3(s3)
        return s3  # (N, 4f, D, H, W)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass producing predictions and projections for both views.

        Returns
        -------
        p1, p2 : torch.Tensor
            Predictions for view 1 and view 2 (gradient flows through these).
        z1, z2 : torch.Tensor
            Projections (used as stop-gradient targets in the SimSiam loss).
        """
        feat1 = self._encode(x1)
        feat2 = self._encode(x2)

        z1 = self.projector(feat1)
        z2 = self.projector(feat2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    @staticmethod
    def loss(
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Negative cosine similarity loss (symmetric)."""
        cos = nn.functional.cosine_similarity

        def _d(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return -cos(p, z, dim=-1).mean()

        return (_d(p1, z2) + _d(p2, z1)) * 0.5


def simsiam(
    n_classes: int = 1,
    in_channels: int = 1,
    projection_dim: int = 2048,
    latent_dim: int = 512,
    weight_decay: float = 0.0005,
    **kwargs,
) -> SimSiam:
    """Factory function for :class:`SimSiam`."""
    return SimSiam(
        n_classes=n_classes,
        in_channels=in_channels,
        projection_dim=projection_dim,
        latent_dim=latent_dim,
        weight_decay=weight_decay,
    )


__all__ = ["SimSiam", "simsiam"]
