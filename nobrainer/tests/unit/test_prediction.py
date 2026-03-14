"""Unit tests for predict() and predict_with_uncertainty()."""

from __future__ import annotations

from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import torch.nn as nn

from nobrainer.prediction import predict, predict_with_uncertainty

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IdentityModel(nn.Module):
    """Minimal 1-class model: sigmoid of a 1×1×1 conv applied to input."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _MultiClassModel(nn.Module):
    """Minimal 3-class model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _make_nifti(shape=(32, 32, 32), tmp_path=None) -> str:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = str(tmp_path / f"vol_{np.random.randint(0, 1e6)}.nii.gz")
    nib.save(img, path)
    return path


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_nifti(self, tmp_path):
        path = _make_nifti((16, 16, 16), tmp_path)
        model = _IdentityModel()
        out = predict(path, model, block_shape=(8, 8, 8), batch_size=2)
        assert isinstance(out, nib.Nifti1Image)

    def test_output_shape_matches_input(self, tmp_path):
        path = _make_nifti((16, 16, 16), tmp_path)
        model = _IdentityModel()
        out = predict(path, model, block_shape=(8, 8, 8), batch_size=2)
        assert out.shape == (16, 16, 16)

    def test_ndarray_input(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        model = _IdentityModel()
        out = predict(arr, model, block_shape=(8, 8, 8), batch_size=2)
        assert out.shape == (16, 16, 16)

    def test_nifti_image_input(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        img = nib.Nifti1Image(arr, np.eye(4))
        model = _IdentityModel()
        out = predict(img, model, block_shape=(8, 8, 8), batch_size=2)
        assert out.shape == (16, 16, 16)

    def test_affine_preserved(self, tmp_path):
        path = _make_nifti((16, 16, 16), tmp_path)
        model = _IdentityModel()
        src_affine = nib.load(path).affine
        out = predict(path, model, block_shape=(8, 8, 8), batch_size=2)
        assert np.allclose(out.affine, src_affine)

    def test_return_probabilities(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        model = _MultiClassModel()
        out = predict(
            arr, model, block_shape=(8, 8, 8), batch_size=2, return_labels=False
        )
        # 3-class probabilities → shape (3, D, H, W)
        assert out.shape[:1] == (3,) or out.ndim == 4

    def test_non_block_aligned_input(self):
        """Volume with shape not divisible by block_shape should still work."""
        arr = np.random.rand(20, 20, 20).astype(np.float32)
        model = _IdentityModel()
        out = predict(arr, model, block_shape=(8, 8, 8), batch_size=2)
        assert out.shape == (20, 20, 20)


# ---------------------------------------------------------------------------
# predict_with_uncertainty()
# ---------------------------------------------------------------------------


class TestPredictWithUncertainty:
    def test_returns_three_niftis(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        model = _IdentityModel()
        label, var, entropy = predict_with_uncertainty(
            arr, model, n_samples=3, block_shape=(8, 8, 8), batch_size=2
        )
        assert isinstance(label, nib.Nifti1Image)
        assert isinstance(var, nib.Nifti1Image)
        assert isinstance(entropy, nib.Nifti1Image)

    def test_output_shapes_match_input(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        model = _IdentityModel()
        label, var, entropy = predict_with_uncertainty(
            arr, model, n_samples=3, block_shape=(8, 8, 8), batch_size=2
        )
        assert label.shape == (16, 16, 16)
        assert var.shape == (16, 16, 16)
        assert entropy.shape == (16, 16, 16)

    def test_variance_nonnegative(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        model = _IdentityModel()
        _, var, _ = predict_with_uncertainty(
            arr, model, n_samples=3, block_shape=(8, 8, 8), batch_size=2
        )
        assert (np.asarray(var.dataobj) >= 0).all()

    def test_entropy_nonnegative(self):
        arr = np.random.rand(16, 16, 16).astype(np.float32)
        model = _IdentityModel()
        _, _, entropy = predict_with_uncertainty(
            arr, model, n_samples=3, block_shape=(8, 8, 8), batch_size=2
        )
        assert (np.asarray(entropy.dataobj) >= 0).all()
