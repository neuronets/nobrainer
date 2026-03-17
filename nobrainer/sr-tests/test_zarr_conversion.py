"""Tests for NIfTI-to-Zarr and Zarr-to-NIfTI conversion."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

zarr = pytest.importorskip("zarr")  # noqa: F841

from nobrainer.io import nifti_to_zarr, zarr_to_nifti  # noqa: E402


def _mgz_to_nifti(mgz_path: str, output_dir: Path) -> Path:
    """Convert .mgz to .nii.gz (niizarr doesn't support MGH)."""
    img = nib.load(mgz_path)
    out = output_dir / (Path(mgz_path).stem + ".nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), img.affine), str(out))
    return out


class TestZarrConversion:
    """Test Zarr round-trip conversion on real brain data."""

    def test_nifti_to_zarr(self, train_eval_split, tmp_path):
        """nifti_to_zarr() creates a valid Zarr store from a real volume."""
        train_data, _ = train_eval_split
        mgz_path = train_data[0][0]

        nii_path = _mgz_to_nifti(mgz_path, tmp_path)
        zarr_path = tmp_path / "brain.zarr"

        result = nifti_to_zarr(nii_path, zarr_path, chunk_shape=(16, 16, 16), levels=1)
        assert Path(result).exists()

        import zarr as zarr_mod

        store = zarr_mod.open_group(str(zarr_path), mode="r")
        assert "0" in store
        arr = np.asarray(store["0"])
        assert arr.ndim == 3

    def test_zarr_to_nifti_roundtrip(self, train_eval_split, tmp_path):
        """zarr_to_nifti() round-trips back to NIfTI with matching shape."""
        train_data, _ = train_eval_split
        mgz_path = train_data[0][0]

        nii_path = _mgz_to_nifti(mgz_path, tmp_path)
        zarr_path = tmp_path / "roundtrip.zarr"
        nifti_to_zarr(nii_path, zarr_path, chunk_shape=(16, 16, 16), levels=1)

        roundtrip_path = tmp_path / "roundtrip.nii.gz"
        zarr_to_nifti(zarr_path, roundtrip_path)

        original = nib.load(str(nii_path))
        roundtrip = nib.load(str(roundtrip_path))
        assert original.shape == roundtrip.shape
        # Value range should be preserved (exact match may differ due to
        # niizarr orientation transforms)
        orig_data = np.asarray(original.dataobj, dtype=np.float32)
        rt_data = np.asarray(roundtrip.dataobj, dtype=np.float32)
        assert abs(orig_data.mean() - rt_data.mean()) < orig_data.std() * 0.5

    def test_multi_resolution_pyramid(self, train_eval_split, tmp_path):
        """nifti_to_zarr(levels=3) creates a multi-resolution pyramid."""
        train_data, _ = train_eval_split
        mgz_path = train_data[0][0]

        nii_path = _mgz_to_nifti(mgz_path, tmp_path)
        zarr_path = tmp_path / "pyramid.zarr"
        nifti_to_zarr(nii_path, zarr_path, chunk_shape=(16, 16, 16), levels=3)

        import zarr as zarr_mod

        store = zarr_mod.open_group(str(zarr_path), mode="r")
        # Should have levels 0, 1, 2
        assert "0" in store
        assert "1" in store
        assert "2" in store

        shape_0 = np.asarray(store["0"]).shape
        shape_1 = np.asarray(store["1"]).shape
        shape_2 = np.asarray(store["2"]).shape

        # Each level should be roughly half the previous
        for dim in range(3):
            assert shape_1[dim] <= shape_0[dim]
            assert shape_2[dim] <= shape_1[dim]
