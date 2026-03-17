"""Tests for Croissant-ML metadata generation."""

import json
from pathlib import Path

from nobrainer.processing import Dataset, Segmentation


class TestCroissantMetadata:
    """Test Croissant-ML provenance in saved models and datasets."""

    def test_segmentation_save_croissant_fields(self, train_eval_split, tmp_path):
        """Segmentation.save() produces croissant.json with provenance fields."""
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

        save_dir = tmp_path / "croissant_model"
        seg.save(save_dir)

        croissant_path = save_dir / "croissant.json"
        assert croissant_path.exists()

        meta = json.loads(croissant_path.read_text())

        # Should have Croissant-ML context or provenance
        has_context = "@context" in meta
        has_provenance = "nobrainer:provenance" in meta
        assert (
            has_context or has_provenance
        ), "croissant.json must have @context or nobrainer:provenance"

        # Check provenance fields if present
        if has_provenance:
            prov = meta["nobrainer:provenance"]
            assert "model_architecture" in prov
            assert "n_classes" in prov
            assert prov["model_architecture"] == "unet"
            assert prov["n_classes"] == 2

    def test_dataset_to_croissant(self, train_eval_split, tmp_path):
        """Dataset.to_croissant() exports dataset metadata."""
        train_data, _ = train_eval_split
        ds = Dataset.from_files(
            train_data,
            block_shape=(16, 16, 16),
            n_classes=2,
        )

        output_path = tmp_path / "dataset_croissant.json"
        result = ds.to_croissant(output_path)

        assert Path(result).exists()
        meta = json.loads(Path(result).read_text())
        assert "@context" in meta or "name" in meta
