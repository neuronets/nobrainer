"""Unit tests for NIfTI <-> Zarr v3 conversion."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="zarr not installed")

from nobrainer.io import nifti_to_zarr, zarr_to_nifti  # noqa: E402


def _make_nifti(tmp_path, shape=(32, 32, 32)):
    """Create a synthetic NIfTI file and return path + data."""
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(data, affine)
    path = str(tmp_path / "test.nii.gz")
    nib.save(img, path)
    return path, data, affine


class TestNiftiToZarr:
    def test_creates_valid_store(self, tmp_path):
        nii_path, data, _ = _make_nifti(tmp_path)
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "out.zarr")
        store = zarr.open_group(str(zarr_path), mode="r")
        arr = np.asarray(store["0"])
        assert arr.shape == data.shape
        assert arr.dtype == np.float32

    def test_provenance_stored(self, tmp_path):
        nii_path, _, _ = _make_nifti(tmp_path)
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "out.zarr")
        store = zarr.open_group(str(zarr_path), mode="r")
        prov = store.attrs.get("nobrainer_provenance")
        assert prov is not None
        assert "source_file" in prov
        assert "created_at" in prov
        assert "nobrainer_version" in prov
        assert prov["tool"] == "nobrainer.io.nifti_to_zarr"

    def test_multi_resolution_pyramid(self, tmp_path):
        nii_path, data, _ = _make_nifti(tmp_path, shape=(64, 64, 64))
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "pyramid.zarr", levels=3)
        store = zarr.open_group(str(zarr_path), mode="r")
        # Level 0: full resolution
        assert np.asarray(store["0"]).shape == (64, 64, 64)
        # Downsampled levels should have smaller shapes
        level1 = np.asarray(store["1"])
        assert all(s <= 64 for s in level1.shape)
        level2 = np.asarray(store["2"])
        assert all(s <= level1.shape[i] for i, s in enumerate(level2.shape))


class TestZarrToNifti:
    def test_round_trip_shape(self, tmp_path):
        """NIfTI -> Zarr -> NIfTI preserves shape."""
        nii_path, data, _ = _make_nifti(tmp_path)
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "rt.zarr")
        rt_path = zarr_to_nifti(zarr_path, tmp_path / "roundtrip.nii.gz")
        rt_img = nib.load(str(rt_path))
        assert rt_img.shape == data.shape

    def test_round_trip_data(self, tmp_path):
        """NIfTI -> Zarr -> NIfTI preserves data values."""
        nii_path, data, _ = _make_nifti(tmp_path)
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "rt.zarr")
        rt_path = zarr_to_nifti(zarr_path, tmp_path / "roundtrip.nii.gz")
        rt_img = nib.load(str(rt_path))
        rt_data = np.asarray(rt_img.dataobj, dtype=np.float32)
        # Value range should be preserved
        assert abs(rt_data.mean() - data.mean()) < 0.1
        assert rt_data.min() >= 0
        assert rt_data.max() <= 1.0 + 0.01

    def test_round_trip_level1(self, tmp_path):
        """Exporting level 1 gives a smaller shape."""
        nii_path, _, _ = _make_nifti(tmp_path, shape=(64, 64, 64))
        zarr_path = nifti_to_zarr(nii_path, tmp_path / "pyr.zarr", levels=2)
        rt_path = zarr_to_nifti(zarr_path, tmp_path / "level1.nii.gz", level=1)
        rt_img = nib.load(str(rt_path))
        # Level 1 should be smaller than full resolution
        assert all(s <= 64 for s in rt_img.shape)
