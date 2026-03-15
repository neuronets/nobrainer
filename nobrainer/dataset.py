"""PyTorch dataset utilities backed by MONAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Spacingd,
)
import numpy as np
import torch


def get_dataset(
    image_paths: list[str | Path],
    label_paths: list[str | Path] | None = None,
    block_shape: tuple[int, int, int] | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    augment: bool = False,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cache_rate: float = 1.0,
    **kwargs: Any,
) -> DataLoader:
    """Build a MONAI-backed :class:`torch.utils.data.DataLoader`.

    Applies the following transform chain:

    ``LoadImaged → EnsureChannelFirstd → Orientationd("RAS")
    → Spacingd(*target_spacing) → NormalizeIntensityd``
    → (if augment) ``RandAffined, RandFlipd, RandGaussianNoised``

    Parameters
    ----------
    image_paths : list
        Paths to input NIfTI volumes.
    label_paths : list or None
        Paths to corresponding label NIfTI volumes.  ``None`` for
        inference-only datasets.
    block_shape : tuple or None
        If provided, spatial patch size ``(D, H, W)`` extracted by MONAI's
        ``RandSpatialCropd``.  ``None`` loads full volumes.
    batch_size : int
        Number of samples per mini-batch.
    num_workers : int
        Number of DataLoader worker processes.
    augment : bool
        Whether to apply random spatial and intensity augmentations.
    target_spacing : tuple of float
        Voxel spacing (mm) to resample volumes to.
    cache_rate : float
        Fraction of dataset to cache in memory (1.0 = all).
    **kwargs
        Additional keyword arguments forwarded to :class:`DataLoader`.

    Returns
    -------
    DataLoader
        PyTorch DataLoader that yields batches of ``{"image": tensor}``
        (or ``{"image": tensor, "label": tensor}`` when labels are given).
    """
    if label_paths is not None and len(image_paths) != len(label_paths):
        raise ValueError(
            f"len(image_paths)={len(image_paths)} != len(label_paths)={len(label_paths)}"
        )

    has_labels = label_paths is not None

    # Build data dicts
    if has_labels:
        data = [
            {"image": str(img), "label": str(lbl)}
            for img, lbl in zip(image_paths, label_paths)
        ]
        keys = ["image", "label"]
    else:
        data = [{"image": str(img)} for img in image_paths]
        keys = ["image"]

    # Core transforms
    transforms: list[Any] = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=target_spacing,
            mode=["bilinear", "nearest"] if has_labels else ["bilinear"],
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    # Optional augmentation
    if augment:
        transforms += [
            RandAffined(
                keys=keys,
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=["bilinear", "nearest"] if has_labels else ["bilinear"],
                padding_mode="border",
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        ]

    if block_shape is not None:
        from monai.transforms import RandSpatialCropd

        transforms.append(
            RandSpatialCropd(keys=keys, roi_size=block_shape, random_size=False)
        )

    dataset = CacheDataset(
        data=data,
        transform=Compose(transforms),
        cache_rate=cache_rate,
        num_workers=max(0, num_workers),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Zarr v3 dataset (requires [zarr] extras)
# ---------------------------------------------------------------------------


class ZarrDataset(torch.utils.data.Dataset):
    """PyTorch Dataset backed by Zarr v3 stores.

    Each item in *data_list* is a dict with ``"image"`` (and optionally
    ``"label"``) keys pointing to ``.zarr`` store paths.
    """

    def __init__(
        self,
        data_list: list[dict[str, str]],
        transform: Any | None = None,
        zarr_level: int = 0,
    ):
        self.data = data_list
        self.transform = transform
        self.level = zarr_level

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        import zarr

        item = self.data[idx]
        store = zarr.open_group(str(item["image"]), mode="r")
        img_arr = np.asarray(store[str(self.level)]).astype(np.float32)

        result: dict[str, Any] = {"image": img_arr[None]}  # add channel dim

        if "label" in item:
            lbl_store = zarr.open_group(str(item["label"]), mode="r")
            lbl_arr = np.asarray(lbl_store[str(self.level)]).astype(np.float32)
            result["label"] = lbl_arr[None]

        if self.transform is not None:
            result = self.transform(result)

        # Convert to tensors if still numpy
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                result[k] = torch.from_numpy(v)

        return result


def _is_zarr_path(path: str | Path) -> bool:
    """Check if a path looks like a Zarr store."""
    return str(path).rstrip("/").endswith(".zarr")


def _get_zarr_dataset(
    data: list[dict[str, str]],
    batch_size: int,
    num_workers: int,
    augment: bool,
    zarr_level: int,
    **kwargs: Any,
) -> DataLoader:
    """Build a DataLoader from Zarr v3 stores."""
    transform = None
    if augment:
        import monai.transforms as mt

        transform = mt.Compose(
            [
                mt.RandAffined(
                    keys=list(data[0].keys()),
                    prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                ),
                mt.RandFlipd(keys=list(data[0].keys()), prob=0.5),
            ]
        )
    dataset = ZarrDataset(data, transform=transform, zarr_level=zarr_level)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )


__all__ = ["get_dataset", "ZarrDataset"]
