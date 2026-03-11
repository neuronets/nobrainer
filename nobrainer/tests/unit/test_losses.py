"""Unit tests for nobrainer.losses (MONAI-backed)."""

import pytest
import torch

import nobrainer.losses as losses_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_pair(batch=2, spatial=16):
    """Return (y_true, y_pred) binary tensors of shape (B, 1, D, H, W)."""
    y_true = torch.randint(0, 2, (batch, 1, spatial, spatial, spatial)).float()
    y_pred = torch.sigmoid(torch.randn(batch, 1, spatial, spatial, spatial))
    return y_true, y_pred


def _multiclass_pair(batch=2, n_classes=3, spatial=8):
    """Return (y_true one-hot, y_pred softmax) tensors."""
    labels = torch.randint(0, n_classes, (batch, spatial, spatial, spatial))
    y_true = torch.zeros(batch, n_classes, spatial, spatial, spatial)
    y_true.scatter_(1, labels.unsqueeze(1), 1.0)
    y_pred = torch.softmax(
        torch.randn(batch, n_classes, spatial, spatial, spatial), dim=1
    )
    return y_true, y_pred


# ---------------------------------------------------------------------------
# dice
# ---------------------------------------------------------------------------


class TestDiceLoss:
    def test_returns_scalar(self):
        y_true, y_pred = _binary_pair()
        loss_fn = losses_module.dice(sigmoid=False)
        loss = loss_fn(y_pred, y_true)
        assert loss.ndim == 0

    def test_non_negative(self):
        y_true, y_pred = _binary_pair()
        loss_fn = losses_module.dice(sigmoid=True)
        loss = loss_fn(y_pred, y_true)
        assert loss.item() >= 0.0

    def test_perfect_prediction_near_zero(self):
        y = torch.ones(1, 1, 8, 8, 8)
        loss_fn = losses_module.dice()
        loss = loss_fn(y, y)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# generalized_dice
# ---------------------------------------------------------------------------


class TestGeneralizedDiceLoss:
    def test_returns_scalar(self):
        y_true, y_pred = _multiclass_pair()
        loss_fn = losses_module.generalized_dice(softmax=False)
        loss = loss_fn(y_pred, y_true)
        assert loss.ndim == 0

    def test_non_negative(self):
        y_true, y_pred = _multiclass_pair()
        loss_fn = losses_module.generalized_dice()
        loss = loss_fn(y_pred, y_true)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


class TestJaccardLoss:
    def test_returns_scalar(self):
        y_true, y_pred = _binary_pair()
        loss_fn = losses_module.jaccard()
        loss = loss_fn(y_pred, y_true)
        assert loss.ndim == 0

    def test_non_negative(self):
        y_true, y_pred = _binary_pair()
        loss_fn = losses_module.jaccard()
        loss = loss_fn(y_pred, y_true)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# tversky
# ---------------------------------------------------------------------------


class TestTverskyLoss:
    def test_returns_scalar(self):
        y_true, y_pred = _binary_pair()
        loss_fn = losses_module.tversky()
        loss = loss_fn(y_pred, y_true)
        assert loss.ndim == 0

    def test_non_negative(self):
        y_true, y_pred = _binary_pair()
        loss_fn = losses_module.tversky(alpha=0.5, beta=0.5)
        loss = loss_fn(y_pred, y_true)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


class TestStubs:
    def test_elbo_returns_tensor(self):
        """elbo() is implemented in Phase 4; non-Bayesian model yields zero KL."""
        import torch.nn as nn

        result = losses_module.elbo(
            nn.Linear(1, 1), kl_weight=1.0, reconstruction_loss=torch.tensor(0.5)
        )
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(0.5)

    def test_wasserstein_raises(self):
        with pytest.raises(NotImplementedError):
            losses_module.wasserstein(torch.zeros(1), torch.zeros(1))


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_known_loss(self):
        fn = losses_module.get("dice")
        assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            losses_module.get("nonexistent")
