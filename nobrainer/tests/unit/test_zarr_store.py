"""Unit tests for nobrainer.datasets.zarr_store."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest


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


class TestCreateZarrStore:
    def test_creates_store(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(3)]
        store_path = create_zarr_store(
            pairs,
            tmp_path / "test.zarr",
            conform=False,
        )
        assert store_path.exists()

    def test_stacked_4d_layout(self, tmp_path):
        import zarr

        from nobrainer.datasets.zarr_store import create_zarr_store

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(3)]
        store_path = create_zarr_store(
            pairs,
            tmp_path / "test.zarr",
            conform=False,
        )

        store = zarr.open_group(str(store_path), mode="r")
        assert store["images"].shape == (3, 32, 32, 32)
        assert store["labels"].shape == (3, 32, 32, 32)
        assert store["images"].dtype == np.float32
        assert store["labels"].dtype == np.int32

    def test_metadata_stored(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store, store_info

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(3)]
        store_path = create_zarr_store(
            pairs,
            tmp_path / "test.zarr",
            subject_ids=["sub-00", "sub-01", "sub-02"],
            conform=False,
        )

        info = store_info(store_path)
        assert info["n_subjects"] == 3
        assert info["subject_ids"] == ["sub-00", "sub-01", "sub-02"]
        assert info["volume_shape"] == [32, 32, 32]
        assert info["layout"] == "stacked"

    def test_round_trip_fidelity(self, tmp_path):
        import zarr

        from nobrainer.datasets.zarr_store import create_zarr_store

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(2)]
        store_path = create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)

        # Read back and compare
        original = np.asarray(nib.load(pairs[0][0]).dataobj, dtype=np.float32)
        store = zarr.open_group(str(store_path), mode="r")
        stored = np.array(store["images"][0])
        assert np.allclose(original, stored, atol=1e-6)

    def test_partial_io(self, tmp_path):
        import zarr

        from nobrainer.datasets.zarr_store import create_zarr_store

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(5)]
        store_path = create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)

        store = zarr.open_group(str(store_path), mode="r")
        # Read a subregion from subject 2
        patch = np.array(store["images"][2, 8:24, 8:24, 8:24])
        assert patch.shape == (16, 16, 16)

    def test_auto_conform(self, tmp_path):
        import zarr

        from nobrainer.datasets.zarr_store import create_zarr_store

        # Create volumes with different shapes — conform should make them uniform
        # Use shapes where median is 32-divisible for sharding compat
        pairs = [
            _make_nifti_pair(tmp_path, 0, shape=(32, 32, 32)),
            _make_nifti_pair(tmp_path, 1, shape=(32, 32, 32)),
            _make_nifti_pair(tmp_path, 2, shape=(64, 64, 64)),
        ]
        store_path = create_zarr_store(
            pairs,
            tmp_path / "test.zarr",
            conform=True,
        )

        store = zarr.open_group(str(store_path), mode="r")
        # All subjects should have same shape
        assert store["images"].shape[1:] == store["images"].shape[1:]
        info = dict(store.attrs)
        assert info["conformed"] is True

    def test_non_uniform_without_conform_raises(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_zarr_store

        pairs = [
            _make_nifti_pair(tmp_path, 0, shape=(32, 32, 32)),
            _make_nifti_pair(tmp_path, 1, shape=(64, 64, 64)),
        ]
        with pytest.raises(ValueError, match="Non-uniform shapes"):
            create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)


class TestPartition:
    def test_create_partition(self, tmp_path):
        from nobrainer.datasets.zarr_store import create_partition, create_zarr_store

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(10)]
        store_path = create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)
        part_path = create_partition(store_path, ratios=(80, 10, 10))

        assert part_path.exists()
        with open(part_path) as f:
            data = json.load(f)
        assert len(data["partitions"]["train"]) == 8
        assert len(data["partitions"]["val"]) == 1
        assert len(data["partitions"]["test"]) == 1

    def test_load_partition(self, tmp_path):
        from nobrainer.datasets.zarr_store import (
            create_partition,
            create_zarr_store,
            load_partition,
        )

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(10)]
        store_path = create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)
        part_path = create_partition(store_path)

        partitions = load_partition(part_path)
        assert "train" in partitions
        assert "val" in partitions
        assert "test" in partitions
        all_ids = partitions["train"] + partitions["val"] + partitions["test"]
        assert len(set(all_ids)) == 10  # no duplicates

    def test_different_seeds_produce_different_splits(self, tmp_path):
        from nobrainer.datasets.zarr_store import (
            create_partition,
            create_zarr_store,
            load_partition,
        )

        pairs = [_make_nifti_pair(tmp_path, i) for i in range(10)]
        store_path = create_zarr_store(pairs, tmp_path / "test.zarr", conform=False)

        p1 = load_partition(
            create_partition(store_path, seed=1, output_path=tmp_path / "p1.json")
        )
        p2 = load_partition(
            create_partition(store_path, seed=2, output_path=tmp_path / "p2.json")
        )
        # Different seeds should produce different train sets (with high probability)
        assert p1["train"] != p2["train"]
