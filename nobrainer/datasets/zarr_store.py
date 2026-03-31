"""Multi-subject Zarr3 dataset store with sharding.

Converts NIfTI collections into a single sharded Zarr3 store where
subjects are stacked along a 4th dimension: ``images[N, D, H, W]``
and ``labels[N, D, H, W]``.  This layout enables efficient partial I/O
for training: reading one subject's patch is a single seek into one
shard file.

Requires the ``[zarr]`` optional extra (``zarr >= 3.0``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _conform_volume(img, target_shape, target_voxel_size=(1.0, 1.0, 1.0)):
    """Conform a nibabel image to target shape and voxel size."""
    from nibabel.processing import conform

    return conform(img, out_shape=target_shape, voxel_size=target_voxel_size)


def _infer_target_shape(
    image_paths: list[str | Path],
    max_scan: int = 50,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Infer target shape and voxel size from input volumes.

    Uses the median shape and modal voxel size across a sample of volumes.
    """
    import nibabel as nib

    shapes = []
    voxel_sizes = []
    for p in image_paths[:max_scan]:
        img = nib.load(p)
        shapes.append(img.shape[:3])
        voxel_sizes.append(tuple(np.abs(img.header.get_zooms()[:3])))

    # Median shape (rounded to nearest integer)
    median_shape = tuple(int(np.median([s[i] for s in shapes])) for i in range(3))

    # Modal voxel size (most common, or median if all different)
    from collections import Counter

    vox_counts = Counter(voxel_sizes)
    if vox_counts:
        modal_voxel = vox_counts.most_common(1)[0][0]
    else:
        modal_voxel = (1.0, 1.0, 1.0)

    return median_shape, modal_voxel


def create_zarr_store(
    image_label_pairs: list[tuple[str, str]],
    output_path: str | Path,
    subject_ids: list[str] | None = None,
    chunk_shape: tuple[int, int, int] = (32, 32, 32),
    shard_shape: tuple[int, int, int] | None = None,
    compressor: str = "blosc",
    conform: bool = True,
    target_shape: tuple[int, int, int] | None = None,
    target_voxel_size: tuple[float, float, float] | None = None,
) -> Path:
    """Convert NIfTI pairs into a single sharded Zarr3 store.

    When ``conform=True`` (default), volumes are conformed to a uniform
    shape so they can be stacked into 4D arrays ``images[N, D, H, W]``
    and ``labels[N, D, H, W]``.  The target shape is inferred from the
    data (median shape) unless explicitly provided.

    Parameters
    ----------
    image_label_pairs : list of (str, str)
        List of ``(image_path, label_path)`` tuples.
    output_path : str or Path
        Output Zarr store directory.
    subject_ids : list of str or None
        Subject identifiers.  If None, auto-generated as ``sub-000``, etc.
    chunk_shape : tuple of int
        Spatial chunk dimensions (default 32³).
    shard_shape : tuple of int or None
        Shard dimensions.  None = auto (full array or large multiple).
    compressor : str
        Compression codec name (default ``"blosc"``).
    conform : bool
        Auto-conform volumes to uniform shape (default True).
    target_shape : tuple of int or None
        Target spatial shape.  None = infer from data.
    target_voxel_size : tuple of float or None
        Target voxel size.  None = infer from data.

    Returns
    -------
    Path
        Path to the created Zarr store.
    """
    import nibabel as nib
    import zarr

    output_path = Path(output_path)
    n_subjects = len(image_label_pairs)

    if subject_ids is None:
        subject_ids = [f"sub-{i:03d}" for i in range(n_subjects)]

    if len(subject_ids) != n_subjects:
        raise ValueError(
            f"subject_ids length ({len(subject_ids)}) != pairs ({n_subjects})"
        )

    image_paths = [p[0] for p in image_label_pairs]

    # Infer or validate target shape
    if conform:
        if target_shape is None or target_voxel_size is None:
            inferred_shape, inferred_voxel = _infer_target_shape(image_paths)
            if target_shape is None:
                target_shape = inferred_shape
            if target_voxel_size is None:
                target_voxel_size = inferred_voxel
            logger.info(
                "Inferred target: shape=%s, voxel_size=%s",
                target_shape,
                target_voxel_size,
            )
    else:
        # Check all shapes are the same
        first_img = nib.load(image_paths[0])
        target_shape = first_img.shape[:3]
        for p in image_paths[1:]:
            img = nib.load(p)
            if img.shape[:3] != target_shape:
                raise ValueError(
                    f"Non-uniform shapes detected ({img.shape[:3]} vs {target_shape}). "
                    "Use conform=True to auto-conform, or ensure all volumes match."
                )

    D, H, W = target_shape
    full_chunk = (1, *chunk_shape)  # one subject per chunk along axis 0

    # Shard shape: group all chunks into a single shard file per array.
    # With chunks=(1,32,32,32) and shards=(N,D,H,W), zarr writes one
    # monolithic shard file containing all chunks — optimal for HPC
    # parallel filesystems that perform poorly with many small files.
    if shard_shape is not None:
        full_shard = (1, *shard_shape)
    else:
        full_shard = (n_subjects, D, H, W)

    # Create store
    store = zarr.open_group(str(output_path), mode="w")

    # Create sharded 4D arrays
    images_arr = store.create_array(
        "images",
        shape=(n_subjects, D, H, W),
        chunks=full_chunk,
        shards=full_shard,
        dtype=np.float32,
    )
    labels_arr = store.create_array(
        "labels",
        shape=(n_subjects, D, H, W),
        chunks=full_chunk,
        shards=full_shard,
        dtype=np.int32,
    )
    logger.info(
        "Created sharded Zarr3: shape=%s, chunks=%s, shards=%s",
        (n_subjects, D, H, W), full_chunk, full_shard,
    )

    # Write volumes
    for i, (img_path, lbl_path) in enumerate(image_label_pairs):
        img = nib.load(img_path)
        lbl = nib.load(lbl_path)

        if conform:
            img = _conform_volume(img, target_shape, target_voxel_size)
            lbl = _conform_volume(lbl, target_shape, target_voxel_size)

        img_data = np.asarray(img.dataobj, dtype=np.float32)
        lbl_data = np.asarray(lbl.dataobj, dtype=np.int32)

        # Ensure shape matches (conforming may produce slightly different shapes)
        if img_data.shape[:3] != target_shape:
            img_data = img_data[:D, :H, :W]
        if lbl_data.shape[:3] != target_shape:
            lbl_data = lbl_data[:D, :H, :W]

        images_arr[i] = img_data[:D, :H, :W]
        labels_arr[i] = lbl_data[:D, :H, :W]

        logger.info("Stored subject %d/%d: %s", i + 1, n_subjects, subject_ids[i])

    # Store metadata
    store.attrs["n_subjects"] = n_subjects
    store.attrs["subject_ids"] = subject_ids
    store.attrs["volume_shape"] = list(target_shape)
    store.attrs["chunk_shape"] = list(chunk_shape)
    store.attrs["layout"] = "stacked"
    store.attrs["image_dtype"] = "float32"
    store.attrs["label_dtype"] = "int32"
    if conform:
        store.attrs["conformed"] = True
        store.attrs["target_shape"] = [int(x) for x in target_shape]
        store.attrs["target_voxel_size"] = [float(x) for x in target_voxel_size]
    else:
        store.attrs["conformed"] = False

    logger.info(
        "Zarr store created: %s (%d subjects, shape=%s)",
        output_path,
        n_subjects,
        target_shape,
    )
    return output_path.resolve()


def store_info(store_path: str | Path) -> dict[str, Any]:
    """Return store metadata without reading voxel data.

    Parameters
    ----------
    store_path : str or Path
        Path to a Zarr store.

    Returns
    -------
    dict
        Store metadata including n_subjects, volume_shape, subject_ids, etc.
    """
    import zarr

    store = zarr.open_group(str(store_path), mode="r")
    return dict(store.attrs)


def create_partition(
    store_path: str | Path,
    ratios: tuple[int, int, int] = (80, 10, 10),
    seed: int = 42,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a partition index JSON file.

    Parameters
    ----------
    store_path : str or Path
        Path to the Zarr store.
    ratios : tuple of int
        (train, val, test) percentages.
    seed : int
        Random seed for reproducibility.
    output_path : str or Path or None
        Output JSON path.  None = ``<store_path>_partition.json``.

    Returns
    -------
    Path
        Path to the written partition JSON file.
    """
    info = store_info(store_path)
    subject_ids = info["subject_ids"]
    n = len(subject_ids)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    total = sum(ratios)
    n_train = int(n * ratios[0] / total)
    n_val = int(n * ratios[1] / total)

    train_ids = [subject_ids[i] for i in indices[:n_train]]
    val_ids = [subject_ids[i] for i in indices[n_train : n_train + n_val]]
    test_ids = [subject_ids[i] for i in indices[n_train + n_val :]]

    partition = {
        "seed": seed,
        "ratios": list(ratios),
        "n_subjects": n,
        "store_path": str(store_path),
        "partitions": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }

    if output_path is None:
        output_path = Path(str(store_path) + "_partition.json")
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(partition, f, indent=2)

    logger.info(
        "Partition created: %s (train=%d, val=%d, test=%d)",
        output_path,
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return output_path


def load_partition(partition_path: str | Path) -> dict[str, list[str]]:
    """Load a partition index and return ``{split: [subject_ids]}``.

    Parameters
    ----------
    partition_path : str or Path
        Path to a partition JSON file.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    with open(partition_path) as f:
        data = json.load(f)
    return data["partitions"]
