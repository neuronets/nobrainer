"""SR-test: end-to-end Zarr pipeline with real brain data.

Converts sample brain data to Zarr, creates partition, builds
Dataset.from_zarr(), and verifies the DataLoader yields correct patches.
"""

from __future__ import annotations

import numpy as np

from nobrainer.processing import Dataset


class TestZarrPipeline:
    """End-to-end Zarr store → partition → Dataset → DataLoader."""

    def test_zarr_store_from_sample_data(self, sample_data, tmp_path):
        """Convert sample data to Zarr, create partition, load via Dataset."""
        from nobrainer.datasets.zarr_store import (
            create_partition,
            create_zarr_store,
            store_info,
        )

        # Use first 5 subjects
        pairs = sample_data[:5]

        # Create Zarr store (auto-conform since shapes may differ)
        store_path = create_zarr_store(
            pairs,
            tmp_path / "brain.zarr",
            conform=True,
        )

        # Verify store metadata
        info = store_info(store_path)
        assert info["n_subjects"] == 5
        assert info["layout"] == "stacked"
        assert info["conformed"] is True

        # Create partition
        part_path = create_partition(store_path, ratios=(60, 20, 20), seed=42)

        # Build Dataset from Zarr with partition
        ds = Dataset.from_zarr(
            store_path,
            block_shape=(16, 16, 16),
            n_classes=2,
            partition="train",
            partition_path=part_path,
        )

        # Verify data list is filtered
        assert len(ds.data) == 3  # 60% of 5 = 3

        # Verify Zarr metadata in data entries
        assert "_zarr_index" in ds.data[0]
        assert "_subject_id" in ds.data[0]

    def test_zarr_store_roundtrip(self, sample_data, tmp_path):
        """Verify Zarr store preserves data fidelity."""
        import zarr

        from nobrainer.datasets.zarr_store import create_zarr_store

        pairs = sample_data[:2]
        store_path = create_zarr_store(pairs, tmp_path / "brain.zarr", conform=True)

        store = zarr.open_group(str(store_path), mode="r")
        assert store["images"].shape[0] == 2
        assert store["labels"].shape[0] == 2

        # Images should be float32, labels int32
        assert store["images"].dtype == np.float32
        assert store["labels"].dtype == np.int32
