"""Predefined augmentation profiles for brain imaging.

Each profile returns a list of MONAI dictionary transforms wrapped with
:class:`~nobrainer.augmentation.transforms.Augmentation` so they are
automatically skipped during inference.

Profiles: ``"none"``, ``"light"``, ``"standard"``, ``"heavy"``.

Usage::

    from nobrainer.augmentation.profiles import get_augmentation_profile

    transforms = get_augmentation_profile("standard", keys=["image", "label"])
"""

from __future__ import annotations

from .transforms import Augmentation


def get_augmentation_profile(
    name: str,
    keys: list[str] | None = None,
) -> list:
    """Return a list of augmentation transforms for the given profile.

    All returned transforms are wrapped with :class:`Augmentation` so
    :class:`TrainableCompose` will skip them during inference.

    Parameters
    ----------
    name : str
        Profile name: ``"none"``, ``"light"``, ``"standard"``, ``"heavy"``.
    keys : list of str or None
        MONAI dictionary keys (default ``["image", "label"]``).

    Returns
    -------
    list
        List of ``Augmentation``-wrapped MONAI transforms.
    """
    from monai.transforms import RandAffined, RandFlipd, RandGaussianNoised

    if keys is None:
        keys = ["image", "label"]
    img_keys = [k for k in keys if k == "image"]
    has_label = "label" in keys
    modes = ["bilinear", "nearest"] if has_label else ["bilinear"]

    if name == "none":
        return []

    if name == "light":
        return [
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=0)),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=1)),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=2)),
        ]

    if name == "standard":
        return [
            Augmentation(
                RandAffined(
                    keys=keys,
                    prob=0.5,
                    rotate_range=(0.15, 0.15, 0.15),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=modes,
                    padding_mode="border",
                )
            ),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=0)),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=1)),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=2)),
            Augmentation(
                RandGaussianNoised(keys=img_keys, prob=0.2, mean=0.0, std=0.1)
            ),
        ]

    if name == "heavy":
        return [
            Augmentation(
                RandAffined(
                    keys=keys,
                    prob=0.8,
                    rotate_range=(0.3, 0.3, 0.3),
                    scale_range=(0.2, 0.2, 0.2),
                    mode=modes,
                    padding_mode="border",
                )
            ),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=0)),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=1)),
            Augmentation(RandFlipd(keys=keys, prob=0.5, spatial_axis=2)),
            Augmentation(
                RandGaussianNoised(keys=img_keys, prob=0.5, mean=0.0, std=0.15)
            ),
        ]

    available = "none, light, standard, heavy"
    raise ValueError(f"Unknown augmentation profile '{name}'. Available: {available}")
