"""Unit tests for unified model forward interface."""

from __future__ import annotations

import torch

from nobrainer.models import get
from nobrainer.models._utils import model_supports_mc


class TestUnifiedForward:
    """All models accept model(x) without error."""

    def test_meshnet(self):
        model = get("meshnet")(n_classes=2, filters=8, receptive_field=37)
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x)
        assert out.shape == (1, 2, 16, 16, 16)

    def test_unet(self):
        model = get("unet")(n_classes=2, channels=(4, 8), strides=(2,))
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x)
        assert out.shape == (1, 2, 16, 16, 16)

    def test_segformer3d(self):
        model = get("segformer3d")(n_classes=2, embed_dims=(16, 32, 80, 128))
        model.eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 32, 32, 32)


class TestMcSupport:
    """Bayesian models support mc parameter."""

    def test_kwyk_meshnet_supports_mc(self):
        model = get("kwyk_meshnet")(n_classes=2, filters=8, receptive_field=37)
        assert model_supports_mc(model)

        x = torch.randn(1, 1, 16, 16, 16)
        out_det = model(x, mc=False)
        out_mc = model(x, mc=True)
        assert out_det.shape == (1, 2, 16, 16, 16)
        assert out_mc.shape == (1, 2, 16, 16, 16)

    def test_bayesian_meshnet_supports_mc(self):
        import pyro

        pyro.clear_param_store()
        model = get("bayesian_meshnet")(n_classes=2, filters=8, receptive_field=37)
        assert model_supports_mc(model)

        x = torch.randn(1, 1, 16, 16, 16)
        with pyro.poutine.trace():
            out = model(x)
        assert out.shape == (1, 2, 16, 16, 16)

    def test_regular_model_no_mc(self):
        model = get("meshnet")(n_classes=2, filters=8, receptive_field=37)
        assert not model_supports_mc(model)

    def test_forward_helper_uses_explicit_check(self):
        """_forward does NOT use try/except TypeError."""
        import inspect

        from nobrainer.prediction import _forward

        source = inspect.getsource(_forward)
        assert "except TypeError" not in source
        assert "model_supports_mc" in source
