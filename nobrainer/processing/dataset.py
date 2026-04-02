"""Fluent Dataset builder for nobrainer estimators."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import zarr

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
        csv_name = _NAMED_MAPPINGS[name_or_path]
        # Primary: inside the nobrainer package (works with pip install)
        pkg_data = Path(__file__).parent.parent / "data" / "label_mappings" / csv_name
        # Fallback: scripts dir (editable installs / development)
        scripts_data = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "kwyk_reproduction"
            / "label_mappings"
            / csv_name
        )
        candidates = [pkg_data, scripts_data]
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

    return _LabelRemap(lookup)


class _LabelRemap:
    """Picklable label remapping callable (needed for DataLoader workers)."""

    def __init__(self, lookup: dict[int, int]):
        self.lookup = lookup

    def __call__(self, x):
        result = torch.zeros_like(x)
        for orig_val, new_val in self.lookup.items():
            result[x == orig_val] = new_val
        return result.long()


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
        self._augment_profile: str = "standard"
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

    @classmethod
    def from_zarr(
        cls,
        store_path: str | Path,
        block_shape: tuple[int, int, int] | None = None,
        n_classes: int = 1,
        partition: str | None = None,
        partition_path: str | Path | None = None,
    ) -> "Dataset":
        """Create a Dataset from a Zarr3 store.

        Parameters
        ----------
        store_path : str or Path
            Path to a Zarr store created by
            :func:`nobrainer.datasets.zarr_store.create_zarr_store`.
        block_shape : tuple or None
            Spatial patch size.
        n_classes : int
            Number of label classes.
        partition : str or None
            Partition to use: ``"train"``, ``"val"``, ``"test"``, or None (all).
        partition_path : str or Path or None
            Path to partition JSON.  If None and partition is set, looks for
            ``<store_path>_partition.json``.
        """
        from nobrainer.datasets.zarr_store import load_partition, store_info

        store_path = Path(store_path)
        info = store_info(store_path)
        subject_ids = info["subject_ids"]
        volume_shape = tuple(info["volume_shape"])

        # Filter by partition
        if partition is not None:
            if partition_path is None:
                partition_path = Path(str(store_path) + "_partition.json")
            parts = load_partition(partition_path)
            if partition not in parts:
                raise ValueError(
                    f"Partition '{partition}' not found. "
                    f"Available: {list(parts.keys())}"
                )
            subject_ids = parts[partition]

        # Build data list referencing zarr indices
        id_to_idx = {sid: i for i, sid in enumerate(info["subject_ids"])}
        data = []
        for sid in subject_ids:
            idx = id_to_idx[sid]
            data.append(
                {
                    "image": f"zarr://{store_path}#images/{idx}",
                    "label": f"zarr://{store_path}#labels/{idx}",
                    "_zarr_store": str(store_path),
                    "_zarr_index": idx,
                    "_subject_id": sid,
                }
            )

        ds = cls(data=data, volume_shape=volume_shape, n_classes=n_classes)
        ds._block_shape = block_shape
        ds._zarr_store_path = str(store_path)
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

    def augment(self, profile: str | bool = True) -> "Dataset":
        """Enable data augmentation.

        Parameters
        ----------
        profile : str or bool
            ``True`` or ``"standard"`` for the standard profile.
            Named profiles: ``"none"``, ``"light"``, ``"standard"``, ``"heavy"``.
            ``False`` disables augmentation.
        """
        if profile is False or profile == "none":
            self._augment = False
        elif profile is True:
            self._augment = True
            self._augment_profile = "standard"
        elif isinstance(profile, str):
            self._augment = True
            self._augment_profile = profile
        self._dataloader = None
        return self

    def mix(
        self,
        generator: "torch.utils.data.Dataset",
        ratio: float = 0.3,
    ) -> "Dataset":
        """Combine this dataset with a synthetic data generator.

        Creates a mixed dataset where each sample is drawn from either
        the real data (this dataset) or the synthetic generator, based
        on the ratio.

        Parameters
        ----------
        generator : torch.utils.data.Dataset
            Synthetic data source (e.g., ``SynthSegGenerator``).
            Must return ``{"image": Tensor, "label": Tensor}`` dicts.
        ratio : float
            Fraction of samples drawn from the generator (default 0.3 = 30%).

        Returns
        -------
        Dataset
            A new Dataset wrapping a ``MixedDataset``.
        """

        mixed = MixedDataset(self, generator, ratio=ratio)
        new_ds = Dataset(
            data=self.data, volume_shape=self.volume_shape, n_classes=self.n_classes
        )
        new_ds._block_shape = self._block_shape
        new_ds._batch_size = self._batch_size
        new_ds._augment = self._augment
        new_ds._augment_profile = self._augment_profile
        new_ds._mixed_dataset = mixed
        new_ds._dataloader = None
        return new_ds

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
            # Build augmentation transforms if enabled
            transforms = None
            if self._augment:
                from monai.transforms import Compose

                from nobrainer.augmentation.profiles import get_augmentation_profile

                aug_transforms = get_augmentation_profile(
                    self._augment_profile, keys=["image", "label"]
                )
                if aug_transforms:
                    transforms = Compose(aug_transforms)

            patch_ds = PatchDataset(
                data=self.data,
                block_shape=self._block_shape or (32, 32, 32),
                patches_per_volume=self._patches_per_volume,
                binarize=self._binarize if self._binarize else None,
                transforms=transforms,
            )
            # Use multiple workers for I/O prefetching — each worker loads
            # patches independently while GPU processes the current batch.
            # Respect SLURM allocation or fall back to cpu_count.
            import os

            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            max_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
            n_workers = max(1, max_cpus - 1)  # leave 1 CPU for main process
            self._dataloader = DataLoader(
                patch_ds,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                num_workers=n_workers,
                prefetch_factor=2,
                persistent_workers=True if n_workers > 0 else False,
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

        # Cache zarr store handles (opened once, reused for all reads)
        self._zarr_cache: dict[str, zarr.Group] = {}

        # Cache volume shapes — use zarr metadata when available (fast)
        self._shapes: list[tuple[int, ...]] = []
        first_parsed = self._parse_zarr_path(str(data[0]["image"])) if data else None
        if first_parsed is not None:
            # All items share the same zarr store — read shape once
            store = self._get_zarr_store(first_parsed[0])
            spatial_shape = store[first_parsed[1]].shape[1:]  # (D, H, W)
            self._shapes = [spatial_shape] * len(data)
        else:
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

        # Read only the patch region (cached zarr handles for speed)
        img_patch = self._read_region_cached(item["image"], slc).astype(np.float32)

        result: dict[str, torch.Tensor] = {
            "image": torch.from_numpy(img_patch[None]),  # add channel dim
        }

        if "label" in item:
            lbl_patch = self._read_region_cached(item["label"], slc).astype(np.float32)
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
            # Remap functions may expect torch tensors (e.g., _load_label_mapping)
            t = torch.from_numpy(lbl.astype(np.int32))
            result = self.binarize(t)
            return result.numpy().astype(np.float32)
        return lbl

    @staticmethod
    def _parse_zarr_path(path: str) -> tuple[str, str, int] | None:
        """Parse zarr://store_path#array_name/subject_index.

        Returns ``(store_path, array_name, subject_index)`` or None.
        """
        if path.startswith("zarr://"):
            rest = path[len("zarr://") :]
            if "#" in rest:
                store_path, fragment = rest.split("#", 1)
                parts = fragment.rsplit("/", 1)
                if len(parts) == 2:
                    return store_path, parts[0], int(parts[1])
                return store_path, fragment, 0
            return rest, "images", 0
        return None

    @staticmethod
    def _get_shape(path: str) -> tuple[int, ...]:
        """Get volume shape without loading full data."""
        path = str(path)
        parsed = PatchDataset._parse_zarr_path(path)
        if parsed is not None:
            import zarr

            store_path, array_name, idx = parsed
            store = zarr.open_group(store_path, mode="r")
            # Shape of the 4D array is (N, D, H, W); return spatial (D, H, W)
            return store[array_name].shape[1:]
        elif path.rstrip("/").endswith(".zarr"):
            import zarr

            store = zarr.open_group(path, mode="r")
            return store["0"].shape
        else:
            import nibabel as nib

            return nib.load(path).shape[:3]

    def _get_zarr_store(self, store_path: str):
        """Get or create a cached zarr group handle."""
        if store_path not in self._zarr_cache:
            import zarr

            self._zarr_cache[store_path] = zarr.open_group(store_path, mode="r")
        return self._zarr_cache[store_path]

    def _read_region_cached(self, path: str, slc: tuple[slice, ...]) -> np.ndarray:
        """Read a spatial region, using cached zarr handles."""
        path = str(path)
        parsed = self._parse_zarr_path(path)
        if parsed is not None:
            store_path, array_name, idx = parsed
            store = self._get_zarr_store(store_path)
            sd, sh, sw = slc
            return np.asarray(store[array_name][idx, sd, sh, sw])
        return self._read_region(path, slc)

    @staticmethod
    def _read_region(path: str, slc: tuple[slice, ...]) -> np.ndarray:
        """Read a spatial region from a volume (static, no caching)."""
        path = str(path)
        parsed = PatchDataset._parse_zarr_path(path)
        if parsed is not None:
            import zarr

            store_path, array_name, idx = parsed
            store = zarr.open_group(store_path, mode="r")
            sd, sh, sw = slc
            return np.asarray(store[array_name][idx, sd, sh, sw])
        elif path.rstrip("/").endswith(".zarr"):
            import zarr

            store = zarr.open_group(path, mode="r")
            return np.asarray(store["0"][slc])
        else:
            import nibabel as nib

            img = nib.load(path)
            return np.asarray(img.dataobj[slc])


class MixedDataset(torch.utils.data.Dataset):
    """Combine a real dataset with a synthetic generator at a given ratio.

    Each ``__getitem__`` call randomly selects from either the real data
    or the generator based on the ratio.

    Parameters
    ----------
    real_dataset : Dataset or torch.utils.data.Dataset
        The real data source.
    generator : torch.utils.data.Dataset
        Synthetic data source (e.g., ``SynthSegGenerator``).
    ratio : float
        Fraction of samples from the generator (0.3 = 30% synthetic).
    """

    def __init__(
        self,
        real_dataset: "Dataset | torch.utils.data.Dataset",
        generator: torch.utils.data.Dataset,
        ratio: float = 0.3,
    ) -> None:
        self.real_dataset = real_dataset
        self.generator = generator
        self.ratio = ratio
        # Total length is the max of real and synthetic
        self._real_len = len(real_dataset) if hasattr(real_dataset, "__len__") else 0
        self._gen_len = len(generator)

    def __len__(self) -> int:
        return max(self._real_len, self._gen_len)

    def __getitem__(self, idx: int) -> dict:
        import random

        if random.random() < self.ratio:
            # Synthetic sample
            gen_idx = idx % self._gen_len
            return self.generator[gen_idx]
        else:
            # Real sample
            real_idx = idx % max(self._real_len, 1)
            if hasattr(self.real_dataset, "dataloader"):
                # Dataset object — use underlying data
                return self.real_dataset.data[real_idx]
            return self.real_dataset[real_idx]
