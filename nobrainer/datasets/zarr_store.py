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


# ---------------------------------------------------------------------------
# Foundational utilities (T003–T007)
# ---------------------------------------------------------------------------


def select_storage_dtype(
    data: np.ndarray,
    mode: str | None = None,
    quantize: bool = False,
) -> dict:
    """Select the most space-efficient storage dtype for an array.

    Parameters
    ----------
    data : ndarray
        Representative data sample (at least one volume).
    mode : str or None
        ``"labels"`` or ``"images"``.  None = auto-detect from dtype
        and unique-value count (integer with ≤ 1000 unique → labels).
    quantize : bool
        If True and mode is images, use int16+scale-factor instead
        of bfloat16 (the default for float images).

    Returns
    -------
    dict
        ``{"dtype": str, "scl_slope": float|None, "scl_inter": float|None,
           "_nobrainer_dtype": str|None}``
    """
    if mode is None:
        if np.issubdtype(data.dtype, np.integer):
            n_unique = len(np.unique(data))
            mode = "labels" if n_unique <= 1000 else "images"
        else:
            mode = "images"

    if mode == "labels":
        n_unique = len(np.unique(data))
        if n_unique <= 256:
            return {
                "dtype": "uint8",
                "scl_slope": None,
                "scl_inter": None,
                "_nobrainer_dtype": None,
            }
        elif n_unique <= 65536:
            return {
                "dtype": "uint16",
                "scl_slope": None,
                "scl_inter": None,
                "_nobrainer_dtype": None,
            }
        else:
            return {
                "dtype": "int32",
                "scl_slope": None,
                "scl_inter": None,
                "_nobrainer_dtype": None,
            }
    else:
        if quantize:
            dmin, dmax = float(data.min()), float(data.max())
            drange = dmax - dmin
            if drange == 0:
                drange = 1.0
            if dmax < 32767 and dmin > -32768:
                slope = drange / 65534.0
                inter = dmin
                return {
                    "dtype": "int16",
                    "scl_slope": slope,
                    "scl_inter": inter,
                    "_nobrainer_dtype": None,
                }
            else:
                slope = drange / 65535.0
                inter = dmin
                return {
                    "dtype": "uint16",
                    "scl_slope": slope,
                    "scl_inter": inter,
                    "_nobrainer_dtype": None,
                }
        else:
            return {
                "dtype": "uint16",
                "scl_slope": None,
                "scl_inter": None,
                "_nobrainer_dtype": "bfloat16",
            }


def suggest_shards(
    n_volumes: int,
    volume_shape: tuple[int, int, int],
    dtype: str = "float32",
    n_input_files: int | None = None,
    levels: int = 1,
) -> dict:
    """Compute optimal whole-subject shard parameters.

    Shard count ≤ 20% of input file count, with log-scale reduction
    for larger datasets.  Whole-subject shards only (no subject splitting).

    Parameters
    ----------
    n_volumes : int
        Number of subjects.
    volume_shape : tuple of int
        Spatial shape ``(D, H, W)``.
    dtype : str
        Array dtype (for byte estimation).
    n_input_files : int or None
        Total input files.  None = ``2 * n_volumes``.
    levels : int
        Pyramid levels (affects inode estimate).

    Returns
    -------
    dict
        ``{"subjects_per_shard", "shard_shape", "n_shards",
           "estimated_inodes", "estimated_shard_bytes"}``
    """
    if n_input_files is None:
        n_input_files = 2 * n_volumes

    itemsize = np.dtype(dtype).itemsize
    vol_bytes = int(np.prod(volume_shape)) * itemsize
    total_bytes = vol_bytes * n_volumes

    # max_shards = floor(0.2 * n_input_files / max(1, log10(total_bytes/1GB)))
    gb = max(total_bytes / 1e9, 0.001)
    log_factor = max(1.0, np.log10(gb))
    max_shards = max(1, int(0.2 * n_input_files / log_factor))
    max_shards = min(max_shards, n_volumes)

    subjects_per_shard = int(np.ceil(n_volumes / max_shards))
    n_shards = int(np.ceil(n_volumes / subjects_per_shard))
    D, H, W = volume_shape
    shard_shape = (subjects_per_shard, D, H, W)
    shard_bytes = subjects_per_shard * vol_bytes
    # 2 arrays (images + labels) × levels
    estimated_inodes = n_shards * 2 * levels

    return {
        "subjects_per_shard": subjects_per_shard,
        "shard_shape": shard_shape,
        "n_shards": n_shards,
        "estimated_inodes": estimated_inodes,
        "estimated_shard_bytes": shard_bytes,
    }


def encode_bfloat16(data: np.ndarray) -> np.ndarray:
    """Convert float32 to bfloat16 stored as uint16.

    Uses PyTorch for the float32→bfloat16 conversion (truncates mantissa),
    then views the result as uint16 for zarr storage.
    """
    import torch

    t = torch.from_numpy(data.astype(np.float32)).to(torch.bfloat16)
    return t.view(torch.uint16).numpy()


def decode_bfloat16(stored: np.ndarray) -> np.ndarray:
    """Convert uint16 (bfloat16 bit pattern) back to float32."""
    import torch

    t = torch.from_numpy(stored.astype(np.uint16)).view(torch.bfloat16)
    return t.float().numpy()


def encode_scale_factor(
    data: np.ndarray,
    target_dtype: str = "int16",
) -> tuple[np.ndarray, float, float]:
    """Encode float data as integer with slope/intercept.

    Returns ``(encoded_array, scl_slope, scl_inter)`` where
    ``original ≈ encoded * scl_slope + scl_inter``.
    """
    dt = np.dtype(target_dtype)
    dmin, dmax = float(data.min()), float(data.max())
    drange = dmax - dmin
    if drange == 0:
        drange = 1.0

    if np.issubdtype(dt, np.signedinteger):
        info = np.iinfo(dt)
        dtype_range = info.max - info.min
        slope = drange / dtype_range
        inter = dmin - info.min * slope
    else:
        info = np.iinfo(dt)
        dtype_range = info.max
        slope = drange / dtype_range
        inter = dmin

    encoded = np.clip((data - inter) / slope, info.min, info.max).astype(dt)
    return encoded, float(slope), float(inter)


def decode_scale_factor(
    stored: np.ndarray,
    scl_slope: float,
    scl_inter: float,
) -> np.ndarray:
    """Decode integer array to float32 using slope/intercept."""
    return stored.astype(np.float32) * scl_slope + scl_inter


def downsample_labels(label_data: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample a discrete label map preserving valid label values.

    Uses max-probability approach (from nifti-zarr): for each label,
    create a binary mask, smooth it with a Gaussian, then at each
    output voxel pick the label whose smoothed probability is highest.

    Parameters
    ----------
    label_data : ndarray
        3-D integer label array.
    factor : int
        Downsampling factor per axis (default 2).

    Returns
    -------
    ndarray
        Downsampled label array with only original label values.
    """
    from scipy.ndimage import gaussian_filter, zoom

    unique_labels = np.unique(label_data)
    target_shape = tuple(s // factor for s in label_data.shape)

    if len(unique_labels) > 500:
        # Too many labels for max-probability — fall back to nearest
        return zoom(label_data.astype(np.float64), 1.0 / factor, order=0).astype(
            label_data.dtype
        )[: target_shape[0], : target_shape[1], : target_shape[2]]

    # Build smoothed probability for each label, pick argmax
    best_prob = np.full(target_shape, -1.0, dtype=np.float64)
    result = np.zeros(target_shape, dtype=label_data.dtype)
    sigma = factor * 0.5

    for lab in unique_labels:
        mask = (label_data == lab).astype(np.float64)
        smoothed = gaussian_filter(mask, sigma=sigma)
        # Downsample the smoothed probability
        down = zoom(smoothed, 1.0 / factor, order=1)
        down = down[: target_shape[0], : target_shape[1], : target_shape[2]]
        better = down > best_prob
        best_prob[better] = down[better]
        result[better] = lab

    return result


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


# ---------------------------------------------------------------------------
# OME-NGFF metadata + pyramid generation (T009–T012)
# ---------------------------------------------------------------------------


def write_ome_metadata(
    group,
    axes: list[str],
    n_levels: int,
    voxel_size: tuple[float, ...] = (1.0, 1.0, 1.0),
    array_prefix: str = "",
) -> None:
    """Write OME-NGFF v0.5 multiscale metadata to a zarr group.

    Parameters
    ----------
    group : zarr.Group
        Zarr group to annotate.
    axes : list of str
        Axis names, e.g. ``["z", "y", "x"]``.
    n_levels : int
        Number of pyramid levels (including full resolution).
    voxel_size : tuple of float
        Physical voxel size at level 0.
    array_prefix : str
        Path prefix for level arrays (e.g. ``"images/"``).
    """
    datasets = []
    for lvl in range(n_levels):
        factor = 2**lvl
        scale = [float(v * factor) for v in voxel_size]
        datasets.append(
            {
                "path": f"{array_prefix}{lvl}",
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )

    multiscales = [
        {
            "version": "0.5",
            "axes": [
                {"name": ax, "type": "space", "unit": "millimeter"} for ax in axes
            ],
            "datasets": datasets,
            "type": "gaussian",
        }
    ]
    group.attrs["multiscales"] = multiscales


def build_pyramid_level(
    base_data: np.ndarray,
    level: int,
    is_labels: bool = False,
) -> np.ndarray:
    """Downsample a 3-D volume by 2^level.

    For images: anti-aliased smooth downsampling via scipy zoom.
    For labels: max-probability label-preserving downsampling.

    Parameters
    ----------
    base_data : ndarray
        3-D array (D, H, W) at full resolution.
    level : int
        Pyramid level (0 = original, 1 = 2× down, etc.).
    is_labels : bool
        Use label-preserving downsampling if True.

    Returns
    -------
    ndarray
        Downsampled array.
    """
    if level == 0:
        return base_data

    factor = 2**level
    if is_labels:
        # Iterative 2× downsampling for better label preservation
        result = base_data
        for _ in range(level):
            result = downsample_labels(result, factor=2)
        return result
    else:
        from scipy.ndimage import zoom

        return zoom(base_data.astype(np.float32), 1.0 / factor, order=1).astype(
            base_data.dtype
        )


def _write_shard_process(shard_spec: tuple) -> int:
    """Write one shard — runs in a forked process.

    Receives a minimal tuple to avoid pickling large objects.
    The forked process inherits the parent's memory (image_label_pairs
    etc.) so no serialization is needed.
    """
    import nibabel as nib
    import zarr

    (
        shard_idx,
        store_path,
        pairs_slice,  # list of (img_path, lbl_path) for this shard only
        target_shape,
        voxel_size,
        do_conform,
        levels,
        level_shapes,
        img_dtype_info,
        img_np_dtype_str,
        lbl_np_dtype_str,
        global_start,
    ) = shard_spec

    store = zarr.open_group(store_path, mode="r+")
    D, H, W = target_shape
    img_np_dtype = np.dtype(img_np_dtype_str)
    lbl_np_dtype = np.dtype(lbl_np_dtype_str)

    def _encode_image(raw_data):
        if img_dtype_info.get("_nobrainer_dtype") == "bfloat16":
            return encode_bfloat16(raw_data)
        elif img_dtype_info.get("scl_slope") is not None:
            encoded, _, _ = encode_scale_factor(raw_data, img_dtype_info["dtype"])
            return encoded
        return raw_data.astype(img_np_dtype)

    for local_i, (img_path, lbl_path) in enumerate(pairs_slice):
        global_i = global_start + local_i
        img = nib.load(img_path)
        lbl = nib.load(lbl_path)
        if do_conform:
            img = _conform_volume(img, target_shape, voxel_size)
            lbl = _conform_volume(lbl, target_shape, voxel_size)
        img_data = np.asarray(img.dataobj, dtype=np.float32)[:D, :H, :W]
        lbl_data = np.asarray(lbl.dataobj, dtype=np.int32)[:D, :H, :W]

        for lvl in range(levels):
            if lvl == 0:
                img_lvl, lbl_lvl = img_data, lbl_data
            else:
                img_lvl = build_pyramid_level(img_data, lvl, is_labels=False)
                lbl_lvl = build_pyramid_level(lbl_data, lvl, is_labels=True)
            ld, lh, lw = level_shapes[lvl]
            store[f"images/{lvl}"][global_i] = _encode_image(img_lvl[:ld, :lh, :lw])
            store[f"labels/{lvl}"][global_i] = lbl_lvl[:ld, :lh, :lw].astype(
                lbl_np_dtype
            )

    return len(pairs_slice)


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
    levels: int = 1,
    quantize: bool = False,
    dtype: str | None = None,
    n_input_files: int | None = None,
) -> Path:
    """Convert NIfTI pairs into a sharded Zarr3 store with optional pyramids.

    When ``conform=True`` (default), volumes are conformed to a uniform
    shape so they can be stacked into 4D arrays.  Pyramid levels are
    generated after conforming (optional, controlled by ``levels``).

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
        Shard dimensions.  None = auto via ``suggest_shards()``.
    compressor : str
        Compression codec name (default ``"blosc"``).
    conform : bool
        Auto-conform volumes to uniform shape (default True).
    target_shape : tuple of int or None
        Target spatial shape.  None = infer from data.
    target_voxel_size : tuple of float or None
        Target voxel size.  None = infer from data.
    levels : int
        Number of pyramid levels (1 = no pyramid, 3 = typical).
    quantize : bool
        Use int16+scale-factor encoding for images (default False).
    dtype : str or None
        Override storage dtype (``"bfloat16"``, ``"int16"``, etc.).
        None = auto-select.
    n_input_files : int or None
        Total input file count for shard heuristic.  None = 2×pairs.

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
        if target_shape is not None:
            # Trust the caller — skip expensive sequential verification
            logger.info("Using provided target_shape=%s (skipping verification)", target_shape)
        else:
            # Infer from first image, verify a sample
            first_img = nib.load(image_paths[0])
            target_shape = first_img.shape[:3]
            # Spot-check a sample instead of every file
            check_indices = np.linspace(0, len(image_paths) - 1, min(10, len(image_paths)), dtype=int)
            for idx in check_indices:
                img = nib.load(image_paths[idx])
                if img.shape[:3] != target_shape:
                    raise ValueError(
                        f"Non-uniform shapes ({img.shape[:3]} vs {target_shape}). "
                        "Use conform=True or ensure all volumes match."
                    )

    D, H, W = target_shape
    voxel_size = target_voxel_size or (1.0, 1.0, 1.0)

    # Determine image storage dtype
    if dtype is not None:
        img_dtype_info = {
            "dtype": dtype if dtype != "bfloat16" else "uint16",
            "scl_slope": None,
            "scl_inter": None,
            "_nobrainer_dtype": "bfloat16" if dtype == "bfloat16" else None,
        }
    else:
        # Auto-select: sample first image for range analysis
        sample_img = nib.load(image_paths[0])
        sample_data = np.asarray(sample_img.dataobj, dtype=np.float32)
        img_dtype_info = select_storage_dtype(
            sample_data, mode="images", quantize=quantize
        )

    # Determine label storage dtype from first label
    sample_lbl = nib.load(image_label_pairs[0][1])
    sample_lbl_data = np.asarray(sample_lbl.dataobj, dtype=np.int32)
    lbl_dtype_info = select_storage_dtype(sample_lbl_data, mode="labels")

    img_np_dtype = np.dtype(img_dtype_info["dtype"])
    lbl_np_dtype = np.dtype(lbl_dtype_info["dtype"])

    # Compute shard shape via heuristic
    if shard_shape is None:
        shard_info = suggest_shards(
            n_subjects,
            target_shape,
            dtype=img_dtype_info["dtype"],
            n_input_files=n_input_files,
            levels=levels,
        )
        full_shard = shard_info["shard_shape"]
        logger.info("Shard heuristic: %s", shard_info)
    else:
        full_shard = shard_shape

    full_chunk = (1, *chunk_shape)

    # Create store
    store = zarr.open_group(str(output_path), mode="w")

    def _create_level_arrays(lvl, lvl_shape):
        """Create image and label arrays for one pyramid level."""
        lvl_D, lvl_H, lvl_W = lvl_shape
        lvl_chunk = (
            1,
            min(chunk_shape[0], lvl_D),
            min(chunk_shape[1], lvl_H),
            min(chunk_shape[2], lvl_W),
        )
        lvl_shard = (full_shard[0], lvl_D, lvl_H, lvl_W)

        img_arr = store.create_array(
            f"images/{lvl}",
            shape=(n_subjects, lvl_D, lvl_H, lvl_W),
            chunks=lvl_chunk,
            shards=lvl_shard,
            dtype=img_np_dtype,
        )
        lbl_arr = store.create_array(
            f"labels/{lvl}",
            shape=(n_subjects, lvl_D, lvl_H, lvl_W),
            chunks=lvl_chunk,
            shards=lvl_shard,
            dtype=lbl_np_dtype,
        )

        # Per-array attributes
        if img_dtype_info["scl_slope"] is not None:
            img_arr.attrs["scl_slope"] = img_dtype_info["scl_slope"]
            img_arr.attrs["scl_inter"] = img_dtype_info["scl_inter"]
        if img_dtype_info["_nobrainer_dtype"]:
            img_arr.attrs["_nobrainer_dtype"] = img_dtype_info["_nobrainer_dtype"]
        img_arr.attrs["interpolation"] = "linear"
        lbl_arr.attrs["interpolation"] = "nearest"

        return img_arr, lbl_arr

    # Create arrays for all levels
    level_shapes = []
    level_arrays = []
    for lvl in range(levels):
        factor = 2**lvl
        lvl_shape = (D // factor, H // factor, W // factor)
        level_shapes.append(lvl_shape)
        level_arrays.append(_create_level_arrays(lvl, lvl_shape))

    n_shards = int(np.ceil(n_subjects / full_shard[0]))
    logger.info(
        "Created Zarr3: %d levels, shape=%s, chunks=%s, shards=%s "
        "(%d shard files per array)",
        levels,
        (n_subjects, D, H, W),
        full_chunk,
        full_shard,
        n_shards,
    )

    # Write all levels — parallel across shards using forked processes
    import concurrent.futures
    import multiprocessing
    import os

    n_workers = min(os.cpu_count() or 1, n_shards, 16)

    # Build per-shard specs — each contains only its own slice of pairs
    shard_specs = []
    for s in range(n_shards):
        start = s * full_shard[0]
        end = min(start + full_shard[0], n_subjects)
        shard_specs.append((
            s,
            str(output_path),
            image_label_pairs[start:end],
            target_shape,
            voxel_size,
            conform,
            levels,
            level_shapes,
            img_dtype_info,
            str(img_np_dtype),
            str(lbl_np_dtype),
            start,
        ))

    logger.info(
        "Writing %d volumes (%d levels) across %d shards with %d workers...",
        n_subjects,
        levels,
        n_shards,
        n_workers,
    )
    # Use fork context — avoids pickling the full argument list
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as pool:
        futures = [
            pool.submit(_write_shard_process, spec) for spec in shard_specs
        ]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += future.result()
            logger.info("Stored %d/%d volumes", done, n_subjects)

    # Write OME-NGFF metadata
    if levels > 1:
        write_ome_metadata(
            store,
            axes=["z", "y", "x"],
            n_levels=levels,
            voxel_size=voxel_size,
            array_prefix="images/",
        )
        write_ome_metadata(
            store,
            axes=["z", "y", "x"],
            n_levels=levels,
            voxel_size=voxel_size,
            array_prefix="labels/",
        )

    # Store nobrainer metadata
    import nobrainer

    store.attrs["n_subjects"] = n_subjects
    store.attrs["subject_ids"] = subject_ids
    store.attrs["volume_shape"] = list(target_shape)
    store.attrs["chunk_shape"] = list(chunk_shape)
    store.attrs["n_levels"] = levels
    store.attrs["layout"] = "stacked"
    store.attrs["image_dtype"] = img_dtype_info["dtype"]
    store.attrs["label_dtype"] = lbl_dtype_info["dtype"]
    store.attrs["nobrainer_version"] = nobrainer.__version__
    store.attrs["conformed"] = bool(conform)
    if conform:
        store.attrs["target_shape"] = [int(x) for x in target_shape]
        store.attrs["target_voxel_size"] = [float(x) for x in voxel_size]
    if img_dtype_info["_nobrainer_dtype"]:
        store.attrs["_nobrainer_dtype"] = img_dtype_info["_nobrainer_dtype"]
    if img_dtype_info["scl_slope"] is not None:
        store.attrs["scl_slope"] = img_dtype_info["scl_slope"]
        store.attrs["scl_inter"] = img_dtype_info["scl_inter"]

    logger.info(
        "Zarr store created: %s (%d subjects, shape=%s, %d levels)",
        output_path,
        n_subjects,
        target_shape,
        levels,
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
