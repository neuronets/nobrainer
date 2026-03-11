"""Unit tests for nobrainer segmentation models (PyTorch)."""

import pytest
import torch

from nobrainer.models import get as get_model
from nobrainer.models.autoencoder import autoencoder
from nobrainer.models.highresnet import highresnet
from nobrainer.models.meshnet import meshnet
from nobrainer.models.segmentation import attention_unet, unet, unetr, vnet
from nobrainer.models.simsiam import simsiam

# Small spatial size to keep tests fast on CPU
SPATIAL = 32
IN_SHAPE = (1, 1, SPATIAL, SPATIAL, SPATIAL)


def _grad_check(model: torch.nn.Module, inp: torch.Tensor) -> bool:
    """Return True if gradients flow through all parameters."""
    model.train()
    out = model(inp)
    if isinstance(out, tuple):
        loss = sum(o.mean() for o in out)
    else:
        loss = out.mean()
    loss.backward()
    return all(p.grad is not None for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# UNet (MONAI)
# ---------------------------------------------------------------------------


class TestUNet:
    def test_output_shape_binary(self):
        m = unet(n_classes=1)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_output_shape_multiclass(self):
        m = unet(n_classes=3)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 3, SPATIAL, SPATIAL, SPATIAL)

    def test_gradient_flow(self):
        m = unet(n_classes=2)
        x = torch.randn(*IN_SHAPE)
        assert _grad_check(m, x)

    def test_get_registry(self):
        fn = get_model("unet")
        assert fn is unet


# ---------------------------------------------------------------------------
# VNet (MONAI)
# ---------------------------------------------------------------------------


class TestVNet:
    def test_output_shape(self):
        m = vnet(n_classes=1)
        x = torch.randn(*IN_SHAPE)
        out = m(x)
        assert out.shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_gradient_flow(self):
        m = vnet(n_classes=2)
        x = torch.randn(*IN_SHAPE)
        assert _grad_check(m, x)


# ---------------------------------------------------------------------------
# Attention UNet (MONAI)
# ---------------------------------------------------------------------------


class TestUNETR:
    def test_output_shape(self):
        m = unetr(
            n_classes=2,
            img_size=(SPATIAL, SPATIAL, SPATIAL),
            hidden_size=192,
            mlp_dim=768,
            num_heads=12,
            feature_size=8,
        )
        x = torch.randn(1, 1, SPATIAL, SPATIAL, SPATIAL)
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 2, SPATIAL, SPATIAL, SPATIAL)


class TestAttentionUNet:
    def test_output_shape(self):
        m = attention_unet(
            n_classes=1,
            channels=(8, 16, 32),
            strides=(2, 2),
        )
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_gradient_flow(self):
        m = attention_unet(
            n_classes=2,
            channels=(8, 16, 32),
            strides=(2, 2),
        )
        x = torch.randn(*IN_SHAPE)
        assert _grad_check(m, x)


# ---------------------------------------------------------------------------
# MeshNet (custom PyTorch)
# ---------------------------------------------------------------------------


class TestMeshNet:
    def test_output_shape_binary(self):
        m = meshnet(n_classes=1)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_output_shape_multiclass(self):
        m = meshnet(n_classes=3)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 3, SPATIAL, SPATIAL, SPATIAL)

    def test_receptive_field_37(self):
        m = meshnet(n_classes=1, receptive_field=37)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_receptive_field_129(self):
        m = meshnet(n_classes=1, receptive_field=129)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_invalid_rf(self):
        with pytest.raises(ValueError, match="receptive_field"):
            meshnet(n_classes=1, receptive_field=999)

    def test_gradient_flow(self):
        m = meshnet(n_classes=2)
        x = torch.randn(*IN_SHAPE)
        assert _grad_check(m, x)


# ---------------------------------------------------------------------------
# HighResNet (custom PyTorch)
# ---------------------------------------------------------------------------


class TestHighResNet:
    def test_output_shape_binary(self):
        m = highresnet(n_classes=1)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 1, SPATIAL, SPATIAL, SPATIAL)

    def test_output_shape_multiclass(self):
        m = highresnet(n_classes=3)
        x = torch.randn(*IN_SHAPE)
        assert m(x).shape == (1, 3, SPATIAL, SPATIAL, SPATIAL)

    def test_gradient_flow(self):
        m = highresnet(n_classes=2)
        x = torch.randn(*IN_SHAPE)
        assert _grad_check(m, x)


# ---------------------------------------------------------------------------
# Autoencoder (custom PyTorch)
# ---------------------------------------------------------------------------


class TestAutoencoder:
    # Use batch=2 to avoid BatchNorm single-sample issues
    def test_output_shape(self):
        m = autoencoder(input_shape=(SPATIAL, SPATIAL, SPATIAL), encoding_dim=64)
        x = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        out = m(x)
        assert out.shape == x.shape

    def test_encode_shape(self):
        m = autoencoder(input_shape=(SPATIAL, SPATIAL, SPATIAL), encoding_dim=64)
        x = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        z = m.encode(x)
        assert z.shape == (2, 64)

    def test_gradient_flow(self):
        m = autoencoder(input_shape=(SPATIAL, SPATIAL, SPATIAL), encoding_dim=32)
        x = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        assert _grad_check(m, x)


# ---------------------------------------------------------------------------
# SimSiam (custom PyTorch)
# ---------------------------------------------------------------------------


class TestSimSiam:
    # Use batch=2 to avoid BatchNorm1d single-sample issues
    def test_forward_shapes(self):
        m = simsiam(projection_dim=128, latent_dim=64)
        x1 = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        x2 = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        p1, p2, z1, z2 = m(x1, x2)
        assert p1.shape == (2, 128)
        assert z1.shape == (2, 128)

    def test_loss_negative_range(self):
        m = simsiam(projection_dim=128, latent_dim=64)
        x1 = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        x2 = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        p1, p2, z1, z2 = m(x1, x2)
        loss = m.loss(p1, p2, z1, z2)
        # Loss should be in [-1, 0] for cosine similarity
        assert -1.1 <= loss.item() <= 0.1

    def test_gradient_flow(self):
        m = simsiam(projection_dim=128, latent_dim=64)
        m.train()
        x1 = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        x2 = torch.randn(2, 1, SPATIAL, SPATIAL, SPATIAL)
        p1, p2, z1, z2 = m(x1, x2)
        loss = m.loss(p1, p2, z1, z2)
        loss.backward()
        assert all(
            p.grad is not None for p in m.projector.parameters() if p.requires_grad
        )
