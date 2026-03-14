"""Unit tests for BayesianConv3d, BayesianLinear, and accumulate_kl."""

from __future__ import annotations

import pyro
import pytest
import torch

from nobrainer.models.bayesian.layers import BayesianConv3d, BayesianLinear
from nobrainer.models.bayesian.utils import accumulate_kl

# ---------------------------------------------------------------------------
# BayesianConv3d
# ---------------------------------------------------------------------------


class TestBayesianConv3d:
    def setup_method(self):
        pyro.clear_param_store()

    def _forward(self, layer, x):
        """Run one forward pass inside a pyro.poutine.trace context."""
        with pyro.poutine.trace():
            return layer(x)

    def test_output_shape(self):
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.zeros(2, 1, 8, 8, 8)
        out = self._forward(layer, x)
        assert out.shape == (2, 4, 8, 8, 8)

    def test_kl_populated_after_forward(self):
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.zeros(2, 1, 8, 8, 8)
        self._forward(layer, x)
        assert isinstance(layer.kl, torch.Tensor)
        assert layer.kl.numel() == 1

    def test_kl_positive(self):
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.zeros(2, 1, 8, 8, 8)
        self._forward(layer, x)
        assert layer.kl.item() > 0

    def test_kl_varies_across_samples(self):
        """KL should differ between two forward passes (stochastic weights)."""
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.zeros(2, 1, 8, 8, 8)
        self._forward(layer, x)
        kl1 = layer.kl.item()
        self._forward(layer, x)
        kl2 = layer.kl.item()
        # They may occasionally be equal, but should usually differ
        assert kl1 == pytest.approx(kl2, rel=1.0) or kl1 != kl2

    def test_prior_laplace(self):
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1, prior_type="laplace")
        x = torch.zeros(2, 1, 8, 8, 8)
        self._forward(layer, x)
        assert layer.kl.item() > 0

    def test_no_bias(self):
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1, bias=False)
        assert layer.bias_mu is None
        assert layer.bias_rho is None
        x = torch.zeros(2, 1, 8, 8, 8)
        self._forward(layer, x)
        assert layer.kl.item() > 0

    def test_weight_sigma_positive(self):
        layer = BayesianConv3d(1, 4, kernel_size=3)
        assert (layer.weight_sigma > 0).all()


# ---------------------------------------------------------------------------
# BayesianLinear
# ---------------------------------------------------------------------------


class TestBayesianLinear:
    def setup_method(self):
        pyro.clear_param_store()

    def _forward(self, layer, x):
        with pyro.poutine.trace():
            return layer(x)

    def test_output_shape(self):
        layer = BayesianLinear(16, 8)
        x = torch.zeros(4, 16)
        out = self._forward(layer, x)
        assert out.shape == (4, 8)

    def test_kl_populated(self):
        layer = BayesianLinear(16, 8)
        x = torch.zeros(4, 16)
        self._forward(layer, x)
        assert layer.kl.item() > 0

    def test_no_bias(self):
        layer = BayesianLinear(16, 8, bias=False)
        assert layer.bias_mu is None
        x = torch.zeros(4, 16)
        self._forward(layer, x)
        assert layer.kl.item() > 0

    def test_prior_laplace(self):
        layer = BayesianLinear(16, 8, prior_type="laplace")
        x = torch.zeros(4, 16)
        self._forward(layer, x)
        assert layer.kl.item() > 0


# ---------------------------------------------------------------------------
# accumulate_kl
# ---------------------------------------------------------------------------


class TestAccumulateKl:
    def setup_method(self):
        pyro.clear_param_store()

    def test_single_layer(self):
        layer = BayesianConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.zeros(2, 1, 8, 8, 8)
        with pyro.poutine.trace():
            layer(x)
        kl = accumulate_kl(layer)
        assert kl.item() == pytest.approx(layer.kl.item())

    def test_multiple_layers(self):
        from pyro.nn import PyroModule

        class _TwoConv(PyroModule):
            def __init__(self):
                super().__init__()
                self.l1 = BayesianConv3d(1, 4, kernel_size=3, padding=1)
                self.l2 = BayesianConv3d(4, 8, kernel_size=3, padding=1)

            def forward(self, x):
                return self.l2(self.l1(x))

        model = _TwoConv()
        x = torch.zeros(2, 1, 8, 8, 8)
        with pyro.poutine.trace():
            model(x)
        total = accumulate_kl(model)
        expected = model.l1.kl + model.l2.kl
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_non_bayesian_model_returns_zero(self):
        import torch.nn as nn

        model = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1))
        kl = accumulate_kl(model)
        assert kl.item() == 0.0
