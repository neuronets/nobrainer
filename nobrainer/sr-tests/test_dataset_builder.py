"""Tests for the fluent Dataset builder with real brain data."""

from nobrainer.processing import Dataset


class TestDatasetBuilder:
    """Test Dataset.from_files() fluent API produces correct outputs."""

    def test_from_files_batch_binarize_augment(self, train_eval_split):
        """Dataset.from_files().batch(2).binarize().augment() produces correct shapes."""
        train_data, _ = train_eval_split
        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
            .augment()
        )
        loader = ds.dataloader
        batch = next(iter(loader))

        assert "image" in batch
        assert "label" in batch
        # batch_size=2, 1 channel, block_shape=(16,16,16)
        assert batch["image"].shape[0] == 2
        assert batch["image"].shape[-3:] == (16, 16, 16)
        assert batch["label"].shape[0] == 2

    def test_split_sizes(self, train_eval_split):
        """Dataset.split() divides data into train/eval with correct sizes."""
        train_data, _ = train_eval_split
        ds = Dataset.from_files(
            train_data,
            block_shape=(16, 16, 16),
            n_classes=2,
        )
        ds_train, ds_eval = ds.split(eval_size=0.2)

        total = len(train_data)
        assert len(ds_train.data) + len(ds_eval.data) == total
        assert len(ds_eval.data) >= 1

    def test_streaming_mode_produces_patches(self, train_eval_split):
        """Dataset.streaming() produces patches via PatchDataset."""
        train_data, _ = train_eval_split
        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
            .streaming(patches_per_volume=2)
        )
        loader = ds.dataloader
        batch = next(iter(loader))

        assert "image" in batch
        assert batch["image"].shape[-3:] == (16, 16, 16)
