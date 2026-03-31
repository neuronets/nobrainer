"""Unit tests for TrainableCompose and Augmentation tagging."""

from __future__ import annotations

from nobrainer.augmentation.transforms import Augmentation, TrainableCompose


def _identity(data):
    """Preprocessing transform that passes data through."""
    data["preprocess_count"] = data.get("preprocess_count", 0) + 1
    return data


def _augment(data):
    """Augmentation transform that modifies data."""
    data["augment_count"] = data.get("augment_count", 0) + 1
    return data


class TestAugmentation:
    def test_wraps_transform(self):
        aug = Augmentation(_augment)
        assert aug.is_augmentation is True
        result = aug({"x": 1})
        assert result["augment_count"] == 1

    def test_repr(self):
        aug = Augmentation(_augment)
        assert "Augmentation" in repr(aug)


class TestTrainableCompose:
    def test_train_mode_runs_all(self):
        pipeline = TrainableCompose([_identity, Augmentation(_augment)])
        result = pipeline({"x": 1}, mode="train")
        assert result["preprocess_count"] == 1
        assert result["augment_count"] == 1

    def test_predict_mode_skips_augmentation(self):
        pipeline = TrainableCompose([_identity, Augmentation(_augment)])
        result = pipeline({"x": 1}, mode="predict")
        assert result["preprocess_count"] == 1
        assert "augment_count" not in result

    def test_default_mode_is_train(self):
        pipeline = TrainableCompose([_identity, Augmentation(_augment)])
        result = pipeline({"x": 1})
        assert result["augment_count"] == 1

    def test_mode_setter(self):
        pipeline = TrainableCompose([_identity, Augmentation(_augment)])
        pipeline.mode = "predict"
        result = pipeline({"x": 1})
        assert "augment_count" not in result

    def test_multiple_augmentations_skipped(self):
        pipeline = TrainableCompose(
            [
                _identity,
                Augmentation(_augment),
                _identity,
                Augmentation(_augment),
            ]
        )
        result = pipeline({"x": 1}, mode="predict")
        assert result["preprocess_count"] == 2
        assert "augment_count" not in result

    def test_train_mode_runs_multiple_augmentations(self):
        pipeline = TrainableCompose(
            [
                _identity,
                Augmentation(_augment),
                _identity,
                Augmentation(_augment),
            ]
        )
        result = pipeline({"x": 1}, mode="train")
        assert result["preprocess_count"] == 2
        assert result["augment_count"] == 2

    def test_empty_pipeline(self):
        pipeline = TrainableCompose([])
        result = pipeline({"x": 1}, mode="train")
        assert result == {"x": 1}


class TestAugmentationProfiles:
    def test_none_returns_empty(self):
        from nobrainer.augmentation.profiles import get_augmentation_profile

        transforms = get_augmentation_profile("none")
        assert transforms == []

    def test_standard_returns_augmentations(self):
        from nobrainer.augmentation.profiles import get_augmentation_profile

        transforms = get_augmentation_profile("standard")
        assert len(transforms) > 0
        assert all(getattr(t, "is_augmentation", False) for t in transforms)

    def test_all_profiles_valid(self):
        from nobrainer.augmentation.profiles import get_augmentation_profile

        for name in ("none", "light", "standard", "heavy"):
            transforms = get_augmentation_profile(name)
            assert isinstance(transforms, list)

    def test_unknown_profile_raises(self):
        import pytest

        from nobrainer.augmentation.profiles import get_augmentation_profile

        with pytest.raises(ValueError, match="Unknown augmentation profile"):
            get_augmentation_profile("extreme")
