"""Tests for the Segmentation estimator with real brain data."""

import json

import nibabel as nib

from nobrainer.processing import Dataset, Segmentation


class TestSegmentationEstimator:
    """Test Segmentation estimator fit/predict/save/load cycle."""

    def test_fit_predict_returns_nifti(self, train_eval_split, tmp_path):
        """Segmentation.fit().predict() returns a NIfTI image."""
        train_data, eval_pair = train_eval_split
        eval_img_path = eval_pair[0]

        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
        )

        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
        )
        seg.fit(ds, epochs=2)
        result = seg.predict(eval_img_path, block_shape=(16, 16, 16))

        assert isinstance(result, nib.Nifti1Image)
        assert len(result.shape) >= 3

    def test_save_creates_croissant(self, train_eval_split, tmp_path):
        """Segmentation.save() creates model.pth and croissant.json."""
        train_data, _ = train_eval_split
        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
        )

        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
        )
        seg.fit(ds, epochs=2)

        save_dir = tmp_path / "saved_model"
        seg.save(save_dir)

        assert (save_dir / "model.pth").exists()
        assert (save_dir / "croissant.json").exists()

        meta = json.loads((save_dir / "croissant.json").read_text())
        assert "@context" in meta or "nobrainer:provenance" in meta

    def test_load_roundtrip(self, train_eval_split, tmp_path):
        """Segmentation.save() then Segmentation.load() restores the model."""
        train_data, eval_pair = train_eval_split
        eval_img_path = eval_pair[0]

        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
        )

        seg = Segmentation(
            "unet",
            model_args={"channels": (4, 8), "strides": (2,)},
        )
        seg.fit(ds, epochs=2)

        save_dir = tmp_path / "roundtrip_model"
        seg.save(save_dir)

        loaded = Segmentation.load(save_dir)
        result = loaded.predict(eval_img_path, block_shape=(16, 16, 16))

        assert isinstance(result, nib.Nifti1Image)
