"""Unit tests for ZarrDataset and get_dataset() Zarr routing."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

zarr = pytest.importorskip("zarr", reason="zarr not installed")

from nobrainer.dataset import ZarrDataset, _is_zarr_path  # noqa: E402
from nobrainer.io import nifti_to_zarr  # noqa: E402


def _make_zarr_pair(tmp_path, shape=(32, 32, 32)):
    """Create a synthetic NIfTI → Zarr pair (image + label)."""
    img_data = np.random.rand(*shape).astype(np.float32)
    lbl_data = (np.random.rand(*shape) > 0.5).astype(np.float32)

    img_nii = tmp_path / "img.nii.gz"
    lbl_nii = tmp_path / "lbl.nii.gz"
    nib.save(nib.Nifti1Image(img_data, np.eye(4)), str(img_nii))
    nib.save(nib.Nifti1Image(lbl_data, np.eye(4)), str(lbl_nii))

    img_zarr = nifti_to_zarr(img_nii, tmp_path / "img.zarr")
    lbl_zarr = nifti_to_zarr(lbl_nii, tmp_path / "lbl.zarr")

    return img_zarr, lbl_zarr, img_data, lbl_data


class TestIsZarrPath:
    def test_zarr_extension(self):
        assert _is_zarr_path("data/brain.zarr")
        assert _is_zarr_path("data/brain.zarr/")

    def test_non_zarr(self):
        assert not _is_zarr_path("data/brain.nii.gz")
        assert not _is_zarr_path("data/brain.h5")


class TestZarrDataset:
    def test_returns_dict_with_image(self, tmp_path):
        img_zarr, _, _, _ = _make_zarr_pair(tmp_path)
        ds = ZarrDataset([{"image": str(img_zarr)}])
        item = ds[0]
        assert "image" in item
        assert isinstance(item["image"], torch.Tensor)

    def test_image_shape_has_channel(self, tmp_path):
        img_zarr, _, img_data, _ = _make_zarr_pair(tmp_path)
        ds = ZarrDataset([{"image": str(img_zarr)}])
        item = ds[0]
        # Should have channel dim: (1, D, H, W)
        assert item["image"].shape == (1, *img_data.shape)

    def test_returns_label_when_provided(self, tmp_path):
        img_zarr, lbl_zarr, _, _ = _make_zarr_pair(tmp_path)
        ds = ZarrDataset([{"image": str(img_zarr), "label": str(lbl_zarr)}])
        item = ds[0]
        assert "label" in item
        assert isinstance(item["label"], torch.Tensor)

    def test_batch_from_dataloader(self, tmp_path):
        img_zarr, lbl_zarr, _, _ = _make_zarr_pair(tmp_path)
        data = [{"image": str(img_zarr), "label": str(lbl_zarr)}]
        ds = ZarrDataset(data)
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        batch = next(iter(loader))
        assert batch["image"].shape[0] == 1  # batch dim
        assert batch["image"].ndim == 5  # (B, C, D, H, W)

    def test_multi_resolution_level(self, tmp_path):
        """Loading at level 1 gives downsampled shape."""
        img_data = np.random.rand(64, 64, 64).astype(np.float32)
        nii_path = tmp_path / "big.nii.gz"
        nib.save(nib.Nifti1Image(img_data, np.eye(4)), str(nii_path))
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "big.zarr", levels=2)

        ds = ZarrDataset([{"image": str(zarr_path)}], zarr_level=1)
        item = ds[0]
        # Level 1 is 2x downsampled: (1, 32, 32, 32)
        assert item["image"].shape == (1, 32, 32, 32)
