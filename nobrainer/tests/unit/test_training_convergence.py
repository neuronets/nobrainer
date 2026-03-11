"""CPU training-convergence smoke tests (US1 acceptance scenario 3).

Verifies that each core segmentation model's training loss at epoch 5 is
lower than at epoch 1 when overfitting a fixed batch on CPU.

Scope: tests nobrainer.models + nobrainer.losses integration; does NOT
require GPU or real data.
"""

from __future__ import annotations

import torch

from nobrainer.losses import dice
from nobrainer.models.highresnet import highresnet
from nobrainer.models.meshnet import meshnet
from nobrainer.models.segmentation import attention_unet, unet, vnet

# Shared synthetic batch: batch_size=2 (satisfies BatchNorm), 32^3 spatial.
# Fixed all-ones label is easy to overfit, keeping the test deterministic.
_SPATIAL = 32
_N_EPOCHS = 5
_LR = 1e-2


def _run_epochs(model: torch.nn.Module, seed: int = 42) -> list[float]:
    """Train *model* for _N_EPOCHS and return per-epoch loss values."""
    torch.manual_seed(seed)
    model.train()
    x = torch.randn(2, 1, _SPATIAL, _SPATIAL, _SPATIAL)
    y = torch.ones(2, 1, _SPATIAL, _SPATIAL, _SPATIAL)
    loss_fn = dice()
    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    losses = []
    for _ in range(_N_EPOCHS):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class TestTrainingConvergence:
    """US1 scenario 3: loss at epoch 5 < loss at epoch 1 for all core models."""

    def test_unet_loss_decreases(self):
        losses = _run_epochs(unet(n_classes=1))
        assert (
            losses[-1] < losses[0]
        ), f"UNet loss did not decrease: epoch1={losses[0]:.4f}, epoch5={losses[-1]:.4f}"

    def test_vnet_loss_decreases(self):
        losses = _run_epochs(vnet(n_classes=1))
        assert (
            losses[-1] < losses[0]
        ), f"VNet loss did not decrease: epoch1={losses[0]:.4f}, epoch5={losses[-1]:.4f}"

    def test_attention_unet_loss_decreases(self):
        losses = _run_epochs(attention_unet(n_classes=1))
        assert (
            losses[-1] < losses[0]
        ), f"AttentionUNet loss did not decrease: epoch1={losses[0]:.4f}, epoch5={losses[-1]:.4f}"

    def test_meshnet_loss_decreases(self):
        losses = _run_epochs(meshnet(n_classes=1))
        assert (
            losses[-1] < losses[0]
        ), f"MeshNet loss did not decrease: epoch1={losses[0]:.4f}, epoch5={losses[-1]:.4f}"

    def test_highresnet_loss_decreases(self):
        losses = _run_epochs(highresnet(n_classes=1))
        assert (
            losses[-1] < losses[0]
        ), f"HighResNet loss did not decrease: epoch1={losses[0]:.4f}, epoch5={losses[-1]:.4f}"
