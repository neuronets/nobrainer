"""Fluent Dataset builder for nobrainer estimators."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


class Dataset:
    """Fluent dataset builder wrapping the nobrainer data pipeline.

    Example::

        ds_train, ds_eval = (
            Dataset.from_files(filepaths, block_shape=(128,128,128))
            .batch(2)
            .augment()
            .normalize()
            .split(eval_size=0.1)
        )
        loader = ds_train.dataloader
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        volume_shape: tuple | None = None,
        n_classes: int = 1,
    ):
        self.data = data
        self.volume_shape = volume_shape
        self.n_classes = n_classes
        self._block_shape: tuple | None = None
        self._batch_size: int = 1
        self._shuffle: bool = False
        self._augment: bool = False
        self._binarize: bool = False
        self._normalizer: Callable | None = None
        self._dataloader: DataLoader | None = None

    @classmethod
    def from_files(
        cls,
        filepaths: list[tuple[str, str]] | list[dict[str, str]],
        block_shape: tuple[int, int, int] | None = None,
        n_classes: int = 1,
    ) -> "Dataset":
        """Create a Dataset from file paths.

        Parameters
        ----------
        filepaths : list
            Either ``[(img, label), ...]`` tuples or
            ``[{"image": img, "label": label}, ...]`` dicts.
        block_shape : tuple or None
            Spatial patch size for extraction. None loads full volumes.
        n_classes : int
            Number of label classes.
        """
        # Normalize to list of dicts
        if filepaths and isinstance(filepaths[0], (list, tuple)):
            data = [{"image": str(img), "label": str(lbl)} for img, lbl in filepaths]
        else:
            data = [{k: str(v) for k, v in d.items()} for d in filepaths]

        # Detect volume shape from first file
        volume_shape = None
        if data:
            import nibabel as nib

            first = data[0]["image"]
            if Path(first).suffix in (".zarr",):
                pass  # Zarr shape detection deferred to dataloader
            else:
                try:
                    volume_shape = nib.load(first).shape[:3]
                except Exception:
                    pass

        ds = cls(data=data, volume_shape=volume_shape, n_classes=n_classes)
        ds._block_shape = block_shape
        return ds

    # --- Fluent API ---

    def batch(self, batch_size: int) -> "Dataset":
        """Set batch size."""
        self._batch_size = batch_size
        self._dataloader = None  # invalidate cache
        return self

    def binarize(self, labels: set[int] | Callable | None = None) -> "Dataset":
        """Binarize or remap labels.

        Parameters
        ----------
        labels : set of ints, callable, or None
            - ``None`` (default): any non-zero value → 1
            - ``set``: voxels with values in the set → 1, all others → 0
            - ``callable``: custom function ``fn(label_tensor) → tensor``

        Examples
        --------
        Brain extraction (any tissue)::

            ds.binarize()

        Select specific FreeSurfer regions (e.g., hippocampus L+R)::

            ds.binarize(labels={17, 53})

        Custom mapping::

            ds.binarize(labels=lambda x: (x >= 1000).float())
        """
        self._binarize = labels if labels is not None else True
        self._dataloader = None
        return self

    def shuffle(self, buffer_size: int = 100) -> "Dataset":
        """Enable shuffling."""
        self._shuffle = True
        self._dataloader = None
        return self

    def augment(self) -> "Dataset":
        """Enable data augmentation."""
        self._augment = True
        self._dataloader = None
        return self

    def normalize(self, fn: Callable | None = None) -> "Dataset":
        """Set normalization function."""
        self._normalizer = fn
        self._dataloader = None
        return self

    def split(self, eval_size: float = 0.1) -> tuple["Dataset", "Dataset"]:
        """Split into train and eval datasets."""
        n = len(self.data)
        n_eval = max(1, int(n * eval_size))
        indices = np.random.permutation(n)
        eval_idx = indices[:n_eval]
        train_idx = indices[n_eval:]

        train_ds = copy.copy(self)
        train_ds.data = [self.data[i] for i in train_idx]
        train_ds._dataloader = None

        eval_ds = copy.copy(self)
        eval_ds.data = [self.data[i] for i in eval_idx]
        eval_ds._dataloader = None

        return train_ds, eval_ds

    @property
    def dataloader(self) -> DataLoader:
        """Lazily build and return a PyTorch DataLoader."""
        if self._dataloader is not None:
            return self._dataloader

        image_paths = [d["image"] for d in self.data]
        label_paths = [d["label"] for d in self.data if "label" in d] or None

        # Check for Zarr paths
        is_zarr = any(str(p).rstrip("/").endswith(".zarr") for p in image_paths)

        if is_zarr:
            from nobrainer.dataset import ZarrDataset

            zarr_data = self.data
            ds = ZarrDataset(zarr_data)
            self._dataloader = DataLoader(
                ds,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            from nobrainer.dataset import get_dataset

            self._dataloader = get_dataset(
                image_paths=image_paths,
                label_paths=label_paths,
                block_shape=self._block_shape,
                batch_size=self._batch_size,
                augment=self._augment,
                binarize_labels=self._binarize,
            )

        return self._dataloader

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def block_shape(self) -> tuple | None:
        return self._block_shape

    def to_croissant(self, output_path: str | Path) -> Path:
        """Export dataset metadata as Croissant-ML JSON-LD."""
        from .croissant import write_dataset_croissant

        return write_dataset_croissant(output_path, self)
