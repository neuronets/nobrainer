"""Unit tests for BayesianVNet and BayesianMeshNet."""

from __future__ import annotations

import pyro
import pytest
import torch

from nobrainer.models.bayesian import (
    BayesianMeshNet,
    BayesianVNet,
    accumulate_kl,
    bayesian_meshnet,
    bayesian_vnet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(model, x):
    """Forward pass inside a Pyro trace context."""
    with pyro.poutine.trace():
        return model(x)


# ---------------------------------------------------------------------------
# BayesianVNet
# ---------------------------------------------------------------------------


class TestBayesianVNet:
    def setup_method(self):
        pyro.clear_param_store()

    def test_default_construction(self):
        m = BayesianVNet()
        assert isinstance(m, BayesianVNet)

    def test_output_shape_single_class(self):
        m = BayesianVNet(n_classes=1, in_channels=1, base_filters=8, levels=2)
        x = torch.zeros(2, 1, 16, 16, 16)
        out = _run(m, x)
        assert out.shape == (2, 1, 16, 16, 16)

    def test_output_shape_multi_class(self):
        m = BayesianVNet(n_classes=4, in_channels=1, base_filters=8, levels=2)
        x = torch.zeros(2, 1, 16, 16, 16)
        out = _run(m, x)
        assert out.shape == (2, 4, 16, 16, 16)

    def test_kl_accumulated(self):
        m = BayesianVNet(n_classes=1, in_channels=1, base_filters=8, levels=2)
        x = torch.zeros(2, 1, 16, 16, 16)
        _run(m, x)
        kl = accumulate_kl(m)
        assert kl.item() > 0

    def test_factory_function(self):
        m = bayesian_vnet(n_classes=2, in_channels=1, base_filters=8, levels=2)
        assert isinstance(m, BayesianVNet)

    def test_laplace_prior(self):
        m = BayesianVNet(
            n_classes=1, in_channels=1, base_filters=8, levels=2, prior_type="laplace"
        )
        x = torch.zeros(2, 1, 16, 16, 16)
        _run(m, x)
        assert accumulate_kl(m).item() > 0

    def test_kl_weight_attribute(self):
        m = BayesianVNet(kl_weight=0.001)
        assert m.kl_weight == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# BayesianMeshNet
# ---------------------------------------------------------------------------


class TestBayesianMeshNet:
    def setup_method(self):
        pyro.clear_param_store()

    def test_default_construction(self):
        m = BayesianMeshNet()
        assert isinstance(m, BayesianMeshNet)

    def test_output_shape_single_class(self):
        m = BayesianMeshNet(n_classes=1, in_channels=1, filters=8, receptive_field=37)
        x = torch.zeros(2, 1, 16, 16, 16)
        out = _run(m, x)
        assert out.shape == (2, 1, 16, 16, 16)

    def test_output_shape_multi_class(self):
        m = BayesianMeshNet(n_classes=4, in_channels=1, filters=8, receptive_field=37)
        x = torch.zeros(2, 1, 16, 16, 16)
        out = _run(m, x)
        assert out.shape == (2, 4, 16, 16, 16)

    def test_kl_accumulated(self):
        m = BayesianMeshNet(n_classes=1, in_channels=1, filters=8, receptive_field=37)
        x = torch.zeros(2, 1, 16, 16, 16)
        _run(m, x)
        assert accumulate_kl(m).item() > 0

    def test_invalid_receptive_field(self):
        with pytest.raises(ValueError, match="receptive_field"):
            BayesianMeshNet(receptive_field=99)

    def test_all_dilation_schedules(self):
        for rf in [37, 67, 129]:
            m = BayesianMeshNet(
                n_classes=1, in_channels=1, filters=4, receptive_field=rf
            )
            x = torch.zeros(2, 1, 8, 8, 8)
            out = _run(m, x)
            assert out.shape == (2, 1, 8, 8, 8)

    def test_factory_function(self):
        m = bayesian_meshnet(n_classes=2, in_channels=1, filters=4, receptive_field=37)
        assert isinstance(m, BayesianMeshNet)

    def test_kl_weight_attribute(self):
        m = BayesianMeshNet(kl_weight=1e-4)
        assert m.kl_weight == pytest.approx(1e-4)
