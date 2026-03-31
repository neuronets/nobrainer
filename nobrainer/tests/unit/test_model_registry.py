"""Unit tests for SwinUNETR and SegResNet model registration."""

from __future__ import annotations

import torch

from nobrainer.models import get


class TestSwinUNETR:
    def test_instantiate(self):
        model = get("swin_unetr")(n_classes=2, feature_size=12)
        assert model is not None

    def test_output_shape(self):
        model = get("swin_unetr")(n_classes=3, feature_size=12)
        model.eval()
        # SwinUNETR needs input >= 64³ due to window attention + instance norm
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 64, 64, 64)


class TestSegResNet:
    def test_instantiate(self):
        model = get("segresnet")(n_classes=2, init_filters=8)
        assert model is not None

    def test_output_shape(self):
        model = get("segresnet")(n_classes=5, init_filters=8, blocks_down=(1, 2, 2, 4))
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (1, 5, 32, 32, 32)


class TestRegistryAccess:
    def test_swin_unetr_in_registry(self):
        from nobrainer.models import available_models

        assert "swin_unetr" in available_models()

    def test_segresnet_in_registry(self):
        from nobrainer.models import available_models

        assert "segresnet" in available_models()
