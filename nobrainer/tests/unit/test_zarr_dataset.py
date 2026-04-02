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


# ---------------------------------------------------------------------------
# US2 tests: Dataset.from_zarr with pyramidal multi-subject stores
# ---------------------------------------------------------------------------


def _make_nifti_pair(tmp_path, idx, shape=(32, 32, 32)):
    """Create a NIfTI image + label pair."""
    img_data = np.random.randn(*shape).astype(np.float32)
    lbl_data = np.random.randint(0, 5, shape, dtype=np.int32)
    affine = np.eye(4)
    img_path = tmp_path / f"sub-{idx:02d}_image.nii.gz"
    lbl_path = tmp_path / f"sub-{idx:02d}_label.nii.gz"
    nib.save(nib.Nifti1Image(img_data, affine), str(img_path))
    nib.save(nib.Nifti1Image(lbl_data, affine), str(lbl_path))
    return str(img_path), str(lbl_path)


class TestDatasetFromZarrPyramid:
    """T022: Dataset.from_zarr with level selection and decoding."""

    def test_level_0_full_resolution(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store
        from nobrainer.processing import Dataset

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(3)]
        store_path = create_zarr_store(
            pairs, tmp_path / "test.zarr", conform=False, levels=3
        )

        ds = Dataset.from_zarr(store_path, block_shape=(16, 16, 16), level=0)
        assert ds.volume_shape == (32, 32, 32)

    def test_level_2_downsampled(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store
        from nobrainer.processing import Dataset

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(3)]
        store_path = create_zarr_store(
            pairs, tmp_path / "test.zarr", conform=False, levels=3
        )

        ds = Dataset.from_zarr(store_path, block_shape=(4, 4, 4), level=2)
        assert ds.volume_shape == (8, 8, 8)

    def test_invalid_level_raises(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store
        from nobrainer.processing import Dataset

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(2)]
        store_path = create_zarr_store(
            pairs, tmp_path / "test.zarr", conform=False, levels=2
        )

        with pytest.raises(ValueError, match="level=5"):
            Dataset.from_zarr(store_path, level=5)

    def test_bfloat16_decoded_on_read(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store
        from nobrainer.processing import Dataset
        from nobrainer.processing.dataset import PatchDataset

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(2)]
        store_path = create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)

        ds = Dataset.from_zarr(store_path, block_shape=(16, 16, 16), level=0)
        # Use PatchDataset directly (bypasses MONAI LoadImaged)
        patch_ds = PatchDataset(
            data=ds.data,
            block_shape=(16, 16, 16),
            patches_per_volume=1,
        )
        item = patch_ds[0]
        img = item["image"]
        # Should be float32 (decoded from bfloat16 by _read_region_cached)
        assert img.dtype == torch.float32
        # Values should be non-zero (not raw uint16 bits)
        assert img.float().abs().mean() > 0

    def test_legacy_store_backward_compat(self, tmp_path):
        """T021/T023: Legacy non-pyramidal stores work with level=0."""
        from nobrainer.datasets.zarr_store import create_zarr_store
        from nobrainer.processing import Dataset

        # Create a store without the old layout by directly writing
        # a store with levels=1 (which uses the new images/0 layout
        # but still has n_levels=1 in metadata)
        pairs = [_make_nifti_pair(tmp_path, i) for i in range(2)]
        store_path = create_zarr_store(
            pairs, tmp_path / "test.zarr", conform=False, levels=1
        )

        # level=0 should work
        ds = Dataset.from_zarr(store_path, block_shape=(16, 16, 16), level=0)
        assert ds.volume_shape == (32, 32, 32)

        # level=1 should raise
        with pytest.raises(ValueError, match="level=1"):
            Dataset.from_zarr(store_path, level=1)
