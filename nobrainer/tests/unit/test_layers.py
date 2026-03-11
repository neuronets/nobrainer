"""Unit tests for nobrainer.layers (PyTorch implementations)."""

import pytest
import torch

from nobrainer.layers import (
    BernoulliDropout,
    ConcreteDropout,
    GaussianDropout,
    MaxPool4D,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHAPE_3D = (1, 1, 8, 8, 8)
SHAPE_4D = (1, 1, 2, 8, 8, 8)  # (N, C, V, D, H, W)


@pytest.fixture
def x3d():
    return torch.ones(SHAPE_3D)


@pytest.fixture
def x4d():
    return torch.ones(SHAPE_4D, requires_grad=False)


# ---------------------------------------------------------------------------
# BernoulliDropout
# ---------------------------------------------------------------------------


class TestBernoulliDropout:
    def test_forward_shape(self, x3d):
        layer = BernoulliDropout(rate=0.3, is_monte_carlo=True)
        layer.train()
        out = layer(x3d)
        assert out.shape == x3d.shape

    def test_passthrough_eval_scale(self, x3d):
        """With scale_during_training=True, eval mode returns x unchanged."""
        layer = BernoulliDropout(
            rate=0.5, is_monte_carlo=False, scale_during_training=True
        )
        layer.eval()
        out = layer(x3d)
        assert torch.allclose(out, x3d)

    def test_passthrough_eval_noscale(self, x3d):
        """With scale_during_training=False, eval mode returns x * keep_prob."""
        rate = 0.3
        layer = BernoulliDropout(
            rate=rate, is_monte_carlo=False, scale_during_training=False
        )
        layer.eval()
        out = layer(x3d)
        assert torch.allclose(out, x3d * (1.0 - rate))

    def test_gradient_flow(self, x3d):
        x = x3d.clone().requires_grad_(True)
        layer = BernoulliDropout(rate=0.3, is_monte_carlo=True, seed=42)
        layer.train()
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            BernoulliDropout(rate=1.0, is_monte_carlo=True)

    def test_mc_applies_in_eval(self, x3d):
        """is_monte_carlo=True applies mask even in eval mode."""
        torch.manual_seed(0)
        layer = BernoulliDropout(rate=0.9, is_monte_carlo=True, seed=1)
        layer.eval()
        out = layer(x3d)
        # With high rate some outputs should be zero
        assert out.sum() < x3d.sum()


# ---------------------------------------------------------------------------
# ConcreteDropout
# ---------------------------------------------------------------------------


class TestConcreteDropout:
    def test_forward_shape(self, x3d):
        N, C, D, H, W = x3d.shape
        layer = ConcreteDropout(in_channels=C, is_monte_carlo=True)
        layer.train()
        out = layer(x3d)
        assert out.shape == x3d.shape

    def test_kl_positive(self, x3d):
        N, C, D, H, W = x3d.shape
        layer = ConcreteDropout(in_channels=C, is_monte_carlo=True)
        layer.train()
        _ = layer(x3d)
        assert layer.kl_loss.item() > 0.0

    def test_gradient_flow(self, x3d):
        N, C, D, H, W = x3d.shape
        x = x3d.clone().requires_grad_(True)
        layer = ConcreteDropout(in_channels=C, is_monte_carlo=True)
        layer.train()
        out = layer(x)
        # Gradient should flow through p_logit (learnable)
        loss = out.sum() + layer.kl_loss
        loss.backward()
        assert layer.p_logit.grad is not None

    def test_p_post_clipped(self, x3d):
        N, C, D, H, W = x3d.shape
        layer = ConcreteDropout(in_channels=C)
        p = layer.p_post
        assert (p >= 0.05).all() and (p <= 0.95).all()

    def test_passthrough_eval(self, x3d):
        N, C, D, H, W = x3d.shape
        layer = ConcreteDropout(
            in_channels=C, is_monte_carlo=False, use_expectation=False
        )
        layer.eval()
        out = layer(x3d)
        assert torch.allclose(out, x3d)


# ---------------------------------------------------------------------------
# GaussianDropout
# ---------------------------------------------------------------------------


class TestGaussianDropout:
    def test_forward_shape(self, x3d):
        layer = GaussianDropout(rate=0.3, is_monte_carlo=True)
        layer.train()
        out = layer(x3d)
        assert out.shape == x3d.shape

    def test_passthrough_eval(self, x3d):
        layer = GaussianDropout(rate=0.3, is_monte_carlo=False)
        layer.eval()
        out = layer(x3d)
        assert torch.allclose(out, x3d)

    def test_gradient_flow(self, x3d):
        x = x3d.clone().requires_grad_(True)
        layer = GaussianDropout(rate=0.3, is_monte_carlo=True, seed=42)
        layer.train()
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_mc_in_eval(self, x3d):
        """is_monte_carlo=True adds noise even in eval mode."""
        torch.manual_seed(0)
        layer = GaussianDropout(rate=0.3, is_monte_carlo=True)
        layer.eval()
        out = layer(x3d)
        # Output should differ from input due to noise
        assert not torch.allclose(out, x3d)

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            GaussianDropout(rate=-0.1, is_monte_carlo=True)


# ---------------------------------------------------------------------------
# MaxPool4D
# ---------------------------------------------------------------------------


class TestMaxPool4D:
    def test_forward_shape(self, x4d):
        layer = MaxPool4D(kernel_size=2, stride=2)
        out = layer(x4d)
        N, C, V, D, H, W = x4d.shape
        assert out.shape == (N, C, V, D // 2, H // 2, W // 2)

    def test_wrong_ndim(self):
        x = torch.ones(1, 1, 8, 8, 8)  # 5-D
        layer = MaxPool4D(kernel_size=2)
        with pytest.raises(ValueError, match="6-D"):
            layer(x)

    def test_pool_v(self):
        x = torch.randn(1, 1, 4, 8, 8, 8)
        layer = MaxPool4D(kernel_size=2, stride=2, pool_v=2)
        out = layer(x)
        assert out.shape[2] == 2  # V reduced from 4 → 2

    def test_gradient_flow(self, x4d):
        x = x4d.clone().float().requires_grad_(True)
        layer = MaxPool4D(kernel_size=2, stride=2)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
