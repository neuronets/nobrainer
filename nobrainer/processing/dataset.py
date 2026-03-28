"""Fluent Dataset builder for nobrainer estimators."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

# Named label mapping CSV locations (relative to package or absolute)
_NAMED_MAPPINGS = {
    "6-class": "6-class-mapping.csv",
    "50-class": "50-class-mapping.csv",
    "115-class": "115-class-mapping.csv",
}


def _load_label_mapping(name_or_path: str) -> Callable:
    """Load a label mapping CSV and return a remap function.

    Accepts named mappings ("6-class", "50-class", "115-class") or a
    path to a CSV with ``original`` and ``new`` columns.
    """
    import csv as csv_mod

    if name_or_path in _NAMED_MAPPINGS:
        # Search for CSV in known locations
        csv_name = _NAMED_MAPPINGS[name_or_path]
        candidates = [
            Path(__file__).parent.parent.parent
            / "scripts"
            / "kwyk_reproduction"
            / "label_mappings"
            / csv_name,
            Path.home()
            / "software"
            / "neuronets"
            / "nobrainer_training_scripts"
            / "csv-files"
            / csv_name,
        ]
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"Label mapping '{name_or_path}' not found. "
                f"Searched: {[str(c) for c in candidates]}"
            )
    else:
        csv_path = Path(name_or_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Label mapping CSV not found: {csv_path}")

    # Parse CSV: build original → new lookup
    lookup = {}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            orig = int(row["original"])
            new = int(row["new"])
            lookup[orig] = new

    def _remap(x):
        """Remap FreeSurfer labels using lookup table."""
        result = torch.zeros_like(x)
        for orig_val, new_val in lookup.items():
            result[x == orig_val] = new_val
        return result.float()

    return _remap


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
        self._streaming: bool = False
        self._patches_per_volume: int = 10
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

    def binarize(self, labels: str | set[int] | Callable | None = None) -> "Dataset":
        """Binarize or remap labels.

        Parameters
        ----------
        labels : str, set of ints, callable, or None
            - ``None`` (default): any non-zero value → 1
            - ``"binary"``: same as None (any non-zero → 1)
            - ``"6-class"``, ``"50-class"``, ``"115-class"``: named
              parcellation from nobrainer_training_scripts mapping CSVs
            - ``set``: voxels with values in the set → 1, all others → 0
            - ``callable``: custom ``fn(label_tensor) → tensor``
            - ``str`` (path): path to a custom mapping CSV with
              ``original,new`` columns

        Examples
        --------
        Brain extraction (any tissue)::

            ds.binarize()

        Named parcellation::

            ds.binarize(labels="50-class")

        Select specific FreeSurfer regions (e.g., hippocampus L+R)::

            ds.binarize(labels={17, 53})

        Custom mapping CSV::

            ds.binarize(labels="/path/to/mapping.csv")
        """
        if isinstance(labels, str) and labels not in ("binary",):
            # Named mapping or CSV path
            self._binarize = _load_label_mapping(labels)
        elif labels is not None:
            self._binarize = labels
        else:
            self._binarize = True
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

    def streaming(self, patches_per_volume: int = 10) -> "Dataset":
        """Use streaming patch extraction (no full-volume loading).

        Instead of loading entire volumes and cropping in memory (MONAI
        pipeline), patches are read directly from disk.  For Zarr stores,
        only the chunks overlapping the requested patch are fetched —
        enabling efficient cloud and large-dataset training.

        Requires ``block_shape`` to be set via ``from_files()`` or
        ``batch()`` first.

        Parameters
        ----------
        patches_per_volume : int
            Random patches per volume per epoch.

        Example
        -------
        ::

            ds = (Dataset.from_files(paths, block_shape=(64,64,64))
                  .batch(4).binarize().streaming(patches_per_volume=20))
        """
        self._streaming = True
        self._patches_per_volume = patches_per_volume
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

        # Streaming mode: use PatchDataset for on-the-fly patch extraction
        if self._streaming:
            patch_ds = PatchDataset(
                data=self.data,
                block_shape=self._block_shape or (32, 32, 32),
                patches_per_volume=self._patches_per_volume,
                binarize=self._binarize if self._binarize else None,
            )
            self._dataloader = DataLoader(
                patch_ds,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
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


def extract_patches(
    volume: np.ndarray,
    label: np.ndarray | None = None,
    block_shape: tuple[int, int, int] = (32, 32, 32),
    n_patches: int = 10,
    binarize: bool | set | Callable | None = None,
) -> list[tuple[np.ndarray, ...]] | list[np.ndarray]:
    """Extract random patches from a 3D volume.

    Parameters
    ----------
    volume : ndarray
        3D volume of shape ``(D, H, W)`` or path loadable by nibabel.
    label : ndarray or None
        Corresponding label volume. If None, only image patches returned.
    block_shape : tuple
        Spatial size of each patch ``(bD, bH, bW)``.
    n_patches : int
        Number of random patches to extract.
    binarize : bool, set, callable, or None
        If not None, applied to label patches:
        - ``True``: any non-zero → 1
        - ``set``: voxels in set → 1
        - ``callable``: custom ``fn(patch) → patch``

    Returns
    -------
    list of tuples ``(image_patch, label_patch)`` if label given,
    or list of ``image_patch`` arrays if label is None.

    Examples
    --------
    ::

        import nibabel as nib
        vol = nib.load("brain.nii.gz").get_fdata()
        lbl = nib.load("label.nii.gz").get_fdata()
        patches = extract_patches(vol, lbl, block_shape=(32, 32, 32), n_patches=20)
        # patches[0] = (image_patch, label_patch), each shape (32, 32, 32)
    """
    import nibabel as nib

    # Load from path if needed
    if isinstance(volume, (str, Path)):
        volume = np.asarray(nib.load(str(volume)).dataobj, dtype=np.float32)
    if isinstance(label, (str, Path)):
        label = np.asarray(nib.load(str(label)).dataobj, dtype=np.float32)

    vol = np.asarray(volume, dtype=np.float32)
    bd, bh, bw = block_shape
    D, H, W = vol.shape[:3]

    patches = []
    for _ in range(n_patches):
        d0 = np.random.randint(0, max(1, D - bd + 1))
        h0 = np.random.randint(0, max(1, H - bh + 1))
        w0 = np.random.randint(0, max(1, W - bw + 1))

        img_patch = vol[d0 : d0 + bd, h0 : h0 + bh, w0 : w0 + bw]

        if label is not None:
            lbl = np.asarray(label, dtype=np.float32)
            lbl_patch = lbl[d0 : d0 + bd, h0 : h0 + bh, w0 : w0 + bw]

            # Apply binarization
            if binarize is True:
                lbl_patch = (lbl_patch > 0).astype(np.float32)
            elif isinstance(binarize, set):
                mask = np.zeros_like(lbl_patch)
                for val in binarize:
                    mask = np.maximum(mask, (lbl_patch == val).astype(np.float32))
                lbl_patch = mask
            elif callable(binarize):
                lbl_patch = binarize(lbl_patch)

            patches.append((img_patch, lbl_patch))
        else:
            patches.append(img_patch)

    return patches


class PatchDataset(torch.utils.data.Dataset):
    """Streaming patch dataset — generates random patches on-the-fly.

    Instead of pre-extracting patches or loading full volumes into memory,
    this dataset lazily reads only the voxels needed for each patch.  For
    Zarr v3 stores, this uses chunk-aligned partial I/O (only the chunks
    overlapping the patch are read from disk/cloud).

    Parameters
    ----------
    data : list of dicts
        ``[{"image": path, "label": path}, ...]``.  Paths can be NIfTI
        (``.nii``, ``.nii.gz``, ``.mgz``) or Zarr (``.zarr``).
    block_shape : tuple
        Spatial size of each patch ``(bD, bH, bW)``.
    patches_per_volume : int
        Number of random patches to yield per volume per epoch.
    binarize : bool, set, callable, or None
        Label remapping (see :func:`extract_patches`).
    transforms : callable or None
        Optional transform applied to each ``(image, label)`` dict after
        extraction (e.g., normalization, augmentation).

    Examples
    --------
    ::

        from nobrainer.processing.dataset import PatchDataset

        ds = PatchDataset(
            data=[{"image": "sub-01.zarr", "label": "sub-01_label.zarr"}],
            block_shape=(64, 64, 64),
            patches_per_volume=10,
            binarize=True,
        )
        loader = DataLoader(ds, batch_size=4, num_workers=2)

    Each epoch yields ``len(data) * patches_per_volume`` patches, with
    different random locations each time.
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        block_shape: tuple[int, int, int] = (32, 32, 32),
        patches_per_volume: int = 10,
        binarize: bool | set | Callable | None = None,
        transforms: Callable | None = None,
    ):
        self.data = data
        self.block_shape = block_shape
        self.patches_per_volume = patches_per_volume
        self.binarize = binarize
        self.transforms = transforms

        # Cache volume shapes to avoid repeated reads
        self._shapes: list[tuple[int, ...]] = []
        for item in data:
            self._shapes.append(self._get_shape(item["image"]))

    def __len__(self) -> int:
        return len(self.data) * self.patches_per_volume

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vol_idx = idx // self.patches_per_volume
        item = self.data[vol_idx]
        shape = self._shapes[vol_idx]

        # Random patch origin
        bd, bh, bw = self.block_shape
        d0 = np.random.randint(0, max(1, shape[0] - bd + 1))
        h0 = np.random.randint(0, max(1, shape[1] - bh + 1))
        w0 = np.random.randint(0, max(1, shape[2] - bw + 1))
        slc = (slice(d0, d0 + bd), slice(h0, h0 + bh), slice(w0, w0 + bw))

        # Read only the patch region
        img_patch = self._read_region(item["image"], slc).astype(np.float32)

        result: dict[str, torch.Tensor] = {
            "image": torch.from_numpy(img_patch[None]),  # add channel dim
        }

        if "label" in item:
            lbl_patch = self._read_region(item["label"], slc).astype(np.float32)
            lbl_patch = self._apply_binarize(lbl_patch)
            result["label"] = torch.from_numpy(lbl_patch[None])

        if self.transforms is not None:
            result = self.transforms(result)

        return result

    def _apply_binarize(self, lbl: np.ndarray) -> np.ndarray:
        """Apply binarization to a label patch."""
        if self.binarize is True:
            return (lbl > 0).astype(np.float32)
        elif isinstance(self.binarize, set):
            mask = np.zeros_like(lbl)
            for val in self.binarize:
                mask = np.maximum(mask, (lbl == val).astype(np.float32))
            return mask
        elif callable(self.binarize):
            return self.binarize(lbl)
        return lbl

    @staticmethod
    def _get_shape(path: str) -> tuple[int, ...]:
        """Get volume shape without loading full data."""
        path = str(path)
        if path.rstrip("/").endswith(".zarr"):
            import zarr

            store = zarr.open_group(path, mode="r")
            return store["0"].shape
        else:
            import nibabel as nib

            return nib.load(path).shape[:3]

    @staticmethod
    def _read_region(path: str, slc: tuple[slice, ...]) -> np.ndarray:
        """Read a spatial region from a volume.

        For Zarr stores, this is a true partial read (only chunks
        overlapping the region are fetched from storage).
        For NIfTI, the full volume header is read but only the
        requested region is loaded into memory via nibabel's
        array proxy.
        """
        path = str(path)
        if path.rstrip("/").endswith(".zarr"):
            import zarr

            store = zarr.open_group(path, mode="r")
            return np.asarray(store["0"][slc])
        else:
            import nibabel as nib

            img = nib.load(path)
            # Use dataobj proxy for memory-efficient slicing
            return np.asarray(img.dataobj[slc])
