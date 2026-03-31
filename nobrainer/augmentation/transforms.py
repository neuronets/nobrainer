"""Augmentation tagging for MONAI transform pipelines.

Extends MONAI's ``Compose`` so individual transforms can be tagged as
**augmentation** (train-only) or **preprocessing** (always runs).  During
inference/prediction, augmentation-tagged transforms are automatically
skipped.

Usage::

    from nobrainer.augmentation.transforms import Augmentation, TrainableCompose
    from monai.transforms import RandAffined, RandGaussianNoised, LoadImaged

    pipeline = TrainableCompose([
        LoadImaged(keys=["image", "label"]),           # preprocessing
        Augmentation(RandAffined(keys=["image", "label"], ...)),  # train-only
        Augmentation(RandGaussianNoised(keys=["image"], ...)),    # train-only
    ])

    # Training: all transforms run
    result = pipeline(data, mode="train")

    # Predict: augmentation transforms are skipped
    result = pipeline(data, mode="predict")
"""

from __future__ import annotations

from typing import Any

from monai.transforms import Compose


class Augmentation:
    """Wrapper that tags a MONAI transform as train-only (augmentation).

    When used inside a :class:`TrainableCompose`, this transform is
    automatically skipped when ``mode="predict"``.

    Parameters
    ----------
    transform : callable
        Any MONAI dictionary transform.
    """

    is_augmentation = True

    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __call__(self, data: Any) -> Any:
        return self.transform(data)

    def __repr__(self) -> str:
        return f"Augmentation({self.transform!r})"


class TrainableCompose(Compose):
    """MONAI Compose that skips augmentation-tagged transforms in predict mode.

    Behaves identically to ``monai.transforms.Compose`` in train mode.
    In predict mode, any transform wrapped with :class:`Augmentation`
    (or having ``is_augmentation = True``) is skipped.

    Parameters
    ----------
    transforms : list
        List of MONAI transforms, optionally wrapped with :class:`Augmentation`.
    mode : str
        Default mode: ``"train"`` or ``"predict"``.  Can be overridden
        per-call via ``__call__(data, mode=...)``.
    """

    def __init__(self, transforms: list, mode: str = "train") -> None:
        super().__init__(transforms)
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("train", "predict"):
            raise ValueError(f"mode must be 'train' or 'predict', got '{value}'")
        self._mode = value

    def __call__(self, data: Any, mode: str | None = None) -> Any:
        """Apply transforms, skipping augmentation in predict mode."""
        active_mode = mode or self._mode

        if active_mode == "train":
            # All transforms run
            return super().__call__(data)

        # Predict mode: skip augmentation transforms
        result = data
        for t in self.transforms:
            if getattr(t, "is_augmentation", False):
                continue
            result = t(result)
        return result
