"""Unit tests for SegFormer3D model."""

from __future__ import annotations

import torch

from nobrainer.models import get
from nobrainer.models.segformer3d import SegFormer3D


class TestSegFormer3DShapes:
    def test_output_shape_32(self):
        model = SegFormer3D(n_classes=2, embed_dims=(16, 32, 80, 128))
        model.eval()
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 32, 32, 32)

    def test_output_shape_64(self):
        model = SegFormer3D(n_classes=5, embed_dims=(16, 32, 80, 128))
        model.eval()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5, 64, 64, 64)

    def test_batch_size_2(self):
        model = SegFormer3D(n_classes=3, embed_dims=(16, 32, 80, 128))
        model.eval()
        x = torch.randn(2, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 32, 32, 32)


class TestSegFormer3DParams:
    def test_default_param_count(self):
        """Default (small) config should have ~4-5M params."""
        model = SegFormer3D(n_classes=50)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 10_000_000  # < 10M

    def test_tiny_param_count(self):
        """Tiny config should have ~1-2M params."""
        model = SegFormer3D(n_classes=50, embed_dims=(16, 32, 80, 128))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 5_000_000  # < 5M

    def test_base_param_count(self):
        """Base config should have ~15-20M params."""
        model = SegFormer3D(n_classes=50, embed_dims=(64, 128, 320, 512))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 10_000_000  # > 10M


class TestSegFormer3DRegistry:
    def test_accessible_via_get(self):
        model = get("segformer3d")(n_classes=2, embed_dims=(16, 32, 80, 128))
        assert model is not None
        assert isinstance(model, SegFormer3D)

    def test_in_available_models(self):
        from nobrainer.models import available_models

        assert "segformer3d" in available_models()

    def test_factory_defaults(self):
        from nobrainer.models.segformer3d import segformer3d

        model = segformer3d(n_classes=2)
        assert isinstance(model, SegFormer3D)
