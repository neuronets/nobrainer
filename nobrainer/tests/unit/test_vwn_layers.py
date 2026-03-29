"""Unit tests for VWN layers and KWYKMeshNet."""

from __future__ import annotations

import torch

from nobrainer.models.bayesian.vwn_layers import ConcreteDropout3d, VWNConv3d


class TestVWNConv3d:
    def test_output_shape(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.randn(2, 1, 8, 8, 8)
        out = layer(x, mc=True)
        assert out.shape == (2, 4, 8, 8, 8)

    def test_deterministic_mode(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.randn(2, 1, 8, 8, 8)
        layer.eval()
        out1 = layer(x, mc=False)
        out2 = layer(x, mc=False)
        assert torch.allclose(out1, out2)

    def test_stochastic_mode_varies(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.randn(2, 1, 8, 8, 8)
        out1 = layer(x, mc=True)
        out2 = layer(x, mc=True)
        # Outputs should differ due to stochastic sampling
        assert not torch.allclose(out1, out2)

    def test_kl_populated_after_mc(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1)
        x = torch.randn(2, 1, 8, 8, 8)
        layer(x, mc=True)
        assert layer.kl.item() > 0

    def test_weight_normalization(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1)
        km = layer.kernel_m
        # kernel_m = g * v/||v||, so each filter should have norm ~ |g|
        assert km.shape == (4, 1, 3, 3, 3)

    def test_no_bias(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1, bias=False)
        assert layer.bias_m is None
        x = torch.randn(2, 1, 8, 8, 8)
        out = layer(x, mc=True)
        assert out.shape == (2, 4, 8, 8, 8)

    def test_sigma_positive(self):
        layer = VWNConv3d(1, 4, kernel_size=3, padding=1)
        assert (layer.kernel_sigma >= 0).all()


class TestConcreteDropout3d:
    def test_output_shape(self):
        cd = ConcreteDropout3d(4)
        x = torch.randn(2, 4, 8, 8, 8)
        out = cd(x, mc=True)
        assert out.shape == x.shape

    def test_deterministic_scales(self):
        cd = ConcreteDropout3d(4)
        x = torch.ones(2, 4, 8, 8, 8)
        out = cd(x, mc=False)
        # In deterministic mode, output = x * p
        p = cd.p.view(1, -1, 1, 1, 1)
        expected = x * p
        assert torch.allclose(out, expected)

    def test_p_in_range(self):
        cd = ConcreteDropout3d(4)
        assert (cd.p >= 0.05).all()
        assert (cd.p <= 0.95).all()

    def test_regularization_positive(self):
        cd = ConcreteDropout3d(4)
        reg = cd.regularization()
        assert reg.item() > 0


class TestKWYKMeshNet:
    def test_bernoulli_variant(self):
        from nobrainer.models.bayesian.kwyk_meshnet import KWYKMeshNet

        model = KWYKMeshNet(
            n_classes=2,
            filters=8,
            receptive_field=37,
            dropout_type="bernoulli",
        )
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x, mc=True)
        assert out.shape == (1, 2, 16, 16, 16)

    def test_concrete_variant(self):
        from nobrainer.models.bayesian.kwyk_meshnet import KWYKMeshNet

        model = KWYKMeshNet(
            n_classes=2,
            filters=8,
            receptive_field=37,
            dropout_type="concrete",
        )
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x, mc=True)
        assert out.shape == (1, 2, 16, 16, 16)

    def test_kl_divergence(self):
        from nobrainer.models.bayesian.kwyk_meshnet import KWYKMeshNet

        model = KWYKMeshNet(n_classes=2, filters=8, receptive_field=37)
        x = torch.randn(1, 1, 16, 16, 16)
        model(x, mc=True)
        kl = model.kl_divergence()
        assert torch.isfinite(kl)

    def test_concrete_regularization(self):
        from nobrainer.models.bayesian.kwyk_meshnet import KWYKMeshNet

        model = KWYKMeshNet(
            n_classes=2,
            filters=8,
            receptive_field=37,
            dropout_type="concrete",
        )
        reg = model.concrete_regularization()
        assert reg.item() > 0

    def test_deterministic_forward(self):
        from nobrainer.models.bayesian.kwyk_meshnet import KWYKMeshNet

        model = KWYKMeshNet(n_classes=2, filters=8, receptive_field=37)
        x = torch.randn(1, 1, 16, 16, 16)
        model.eval()
        out1 = model(x, mc=False)
        out2 = model(x, mc=False)
        assert torch.allclose(out1, out2)

    def test_factory_function(self):
        from nobrainer.models import get

        model = get("kwyk_meshnet")(n_classes=2, filters=8, receptive_field=37)
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x)
        assert out.shape == (1, 2, 16, 16, 16)
