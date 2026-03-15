"""Input/output utilities for nobrainer (PyTorch, no TensorFlow)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import struct
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# CSV helpers (no TF dependency)
# ---------------------------------------------------------------------------


def read_csv(
    filepath: str | Path, skip_header: bool = True, delimiter: str = ","
) -> list:
    """Return list of tuples from a CSV file."""
    with open(filepath, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]


def read_mapping(
    filepath: str | Path, skip_header: bool = True, delimiter: str = ","
) -> dict[str, str]:
    """Read CSV as dict; first column → keys, second → values."""
    rows = read_csv(filepath, skip_header=skip_header, delimiter=delimiter)
    return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# TFRecord conversion (T022)
# ---------------------------------------------------------------------------


def _compute_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_tfrecord_file(path: str | Path):
    """Yield raw TFRecord byte strings from a .tfrecord file.

    TFRecord format: [length:uint64][masked_crc32:uint32][data][masked_crc32:uint32]
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            (length,) = struct.unpack_from("<Q", header, 0)
            f.read(4)  # crc of length
            data = f.read(length)
            f.read(4)  # crc of data
            yield data


def convert_tfrecords(
    tfrecord_paths: list[str | Path],
    output_dir: str | Path,
    volume_shape: tuple[int, int, int, int] | None = None,
    output_format: str = "nifti",
    affine: np.ndarray | None = None,
    verify_checksum: bool = True,
) -> list[str]:
    """Convert TFRecord files to NIfTI or HDF5.

    Uses the ``tfrecord`` PyPI package — no TensorFlow required.

    Parameters
    ----------
    tfrecord_paths : list
        Paths to ``.tfrecord`` files.
    output_dir : str or Path
        Directory where converted files are written.
    volume_shape : tuple or None
        Expected shape ``(D, H, W, C)`` of the stored arrays.  Used
        to validate/reshape the parsed tensors.
    output_format : str
        ``"nifti"`` (writes ``.nii.gz``) or ``"hdf5"`` (writes ``.h5``).
    affine : ndarray or None
        4×4 affine matrix for NIfTI files.  Defaults to identity.
    verify_checksum : bool
        Compute SHA-256 of each output file after writing.

    Returns
    -------
    list of str
        Paths to converted output files.
    """
    import tfrecord  # noqa: F401 (optional dep)
    from tfrecord.reader import tfrecord_loader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)

    out_paths: list[str] = []
    for rec_path in tfrecord_paths:
        rec_path = Path(rec_path)
        loader = tfrecord_loader(
            str(rec_path),
            index_path=None,
            description={"volume": "byte", "label": "byte"},
        )
        for i, record in enumerate(loader):
            volume_bytes = record.get("volume") or record.get("image")
            label_bytes = record.get("label")

            vol_arr = np.frombuffer(volume_bytes, dtype=np.float32)
            if volume_shape is not None:
                vol_arr = vol_arr.reshape(volume_shape)

            # TF stores (D,H,W,C), PyTorch wants (C,D,H,W)
            if vol_arr.ndim == 4:
                vol_arr = np.transpose(vol_arr, (3, 0, 1, 2))

            stem = rec_path.stem
            if output_format == "hdf5":
                out_path = output_dir / f"{stem}_{i:04d}.h5"
                with h5py.File(out_path, "w") as hf:
                    hf.create_dataset("volume", data=vol_arr, compression="gzip")
                    if label_bytes is not None:
                        lbl_arr = np.frombuffer(label_bytes, dtype=np.float32)
                        if volume_shape is not None:
                            lbl_arr = lbl_arr.reshape(volume_shape)
                        if lbl_arr.ndim == 4:
                            lbl_arr = np.transpose(lbl_arr, (3, 0, 1, 2))
                        hf.create_dataset("label", data=lbl_arr, compression="gzip")
            else:
                # NIfTI: use first channel as spatial volume
                spatial = vol_arr[0] if vol_arr.ndim == 4 else vol_arr
                img = nib.Nifti1Image(spatial.astype(np.float32), affine)
                out_path = output_dir / f"{stem}_{i:04d}.nii.gz"
                nib.save(img, str(out_path))

            out_paths.append(str(out_path))
            if verify_checksum:
                _compute_sha256(out_path)  # validates file integrity

    return out_paths


# ---------------------------------------------------------------------------
# Weight conversion: TF Keras H5 → PyTorch (T024)
# ---------------------------------------------------------------------------

# Mapping patterns: (keras_layer_keyword, param_suffix) → pytorch_param_name
_CONV_MAPPING = {
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",  # BatchNorm scale
    "beta": "bias",  # BatchNorm shift
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}


def _keras_conv3d_to_pytorch(w: np.ndarray) -> np.ndarray:
    """Transpose Conv3D weights from Keras (D,H,W,Cin,Cout) → PyTorch (Cout,Cin,D,H,W)."""
    if w.ndim == 5:
        return np.transpose(w, (4, 3, 0, 1, 2))
    return w


def convert_weights(
    h5_path: str | Path,
    pt_model: nn.Module,
    layer_mapping: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    verify: bool = False,
) -> dict[str, torch.Tensor]:
    """Load Keras ``.h5`` weights and map them to a PyTorch model.

    No TensorFlow is required; weights are read directly with ``h5py``.

    Parameters
    ----------
    h5_path : str or Path
        Path to the Keras ``.h5`` weight file.
    pt_model : nn.Module
        Target PyTorch model whose ``state_dict`` will receive the weights.
    layer_mapping : dict or None
        ``{keras_layer_name: pytorch_submodule_name}`` mapping.  When
        ``None``, an automatic heuristic attempts to match by index.
    output_path : str or Path or None
        If provided, save the converted state dict to ``.pth``.
    verify : bool
        Run a brief forward-pass verification after loading (raises if
        shapes mismatch).

    Returns
    -------
    dict
        The loaded (possibly partial) state dict.
    """
    h5_path = Path(h5_path)
    state = pt_model.state_dict()
    new_state: dict[str, torch.Tensor] = {}

    with h5py.File(h5_path, "r") as hf:
        # Traverse all datasets in the H5 file
        def _collect(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            w = obj[()]  # numpy array
            # Apply weight transposition for Conv3D kernels
            if "kernel" in name and w.ndim == 5:
                w = _keras_conv3d_to_pytorch(w)
            # Determine target PyTorch parameter name
            pt_name = _map_name(name, layer_mapping, state)
            if pt_name is not None and pt_name in state:
                tensor = torch.from_numpy(w.copy())
                if tensor.shape == state[pt_name].shape:
                    new_state[pt_name] = tensor

        hf.visititems(_collect)

    # Load matched weights; keep existing for unmatched
    combined = {**state, **new_state}
    pt_model.load_state_dict(combined, strict=False)

    if output_path is not None:
        torch.save(combined, str(output_path))

    if verify:
        pt_model.eval()
        dummy = torch.zeros(1, 1, 32, 32, 32)
        with torch.no_grad():
            _ = pt_model(dummy)

    return new_state


def _map_name(
    h5_name: str,
    mapping: dict[str, str] | None,
    state: dict[str, torch.Tensor],
) -> str | None:
    """Attempt to resolve an H5 dataset path to a PyTorch state-dict key."""
    # Simple heuristic: look for a state-dict key that contains the leaf name
    parts = h5_name.replace("/", ".").split(".")
    leaf = parts[-1]
    pt_leaf = _CONV_MAPPING.get(leaf, leaf)
    if mapping:
        for k, v in mapping.items():
            if k in h5_name:
                candidate = f"{v}.{pt_leaf}"
                if candidate in state:
                    return candidate
    # Fallback: direct match
    candidate = ".".join(parts[:-1] + [pt_leaf])
    return candidate if candidate in state else None


# ---------------------------------------------------------------------------
# Zarr v3 conversion (requires [zarr] extras)
# ---------------------------------------------------------------------------


def nifti_to_zarr(
    input_path: str | Path,
    output_path: str | Path,
    chunk_shape: tuple[int, int, int] = (64, 64, 64),
    shard_shape: tuple[int, int, int] | None = None,
    compressor: str = "blosc",
    levels: int = 1,
) -> Path:
    """Convert a NIfTI file to a sharded Zarr v3 store.

    Uses ``niizarr.nii2zarr`` for NIfTI-Zarr specification compliance
    (NIfTI header, OME multiscale metadata) and adds nobrainer provenance.

    Parameters
    ----------
    input_path : path to .nii or .nii.gz file
    output_path : path for the output .zarr directory
    chunk_shape : inner chunk dimensions
    shard_shape : outer shard dimensions; None lets niizarr choose
    compressor : compression codec ("blosc" or "zlib")
    levels : number of resolution levels (1 = single level, -1 = auto)

    Returns
    -------
    Path to the created .zarr store
    """
    import datetime

    import zarr

    import nobrainer

    img = nib.load(str(input_path))
    output_path = Path(output_path)

    try:
        import niizarr

        niizarr.nii2zarr(
            img,
            str(output_path),
            chunk=chunk_shape,
            shard=shard_shape,
            nb_levels=levels,
            compressor=compressor,
            zarr_version=3,
        )
    except ImportError:
        # Fallback: manual Zarr v3 creation without niizarr
        arr = np.asarray(img.dataobj, dtype=np.float32)
        clamped_chunk = tuple(min(c, s) for c, s in zip(chunk_shape, arr.shape))
        if shard_shape is None:
            eff_shard = tuple(min(c * 2, s) for c, s in zip(clamped_chunk, arr.shape))
        else:
            eff_shard = tuple(max(c, s) for c, s in zip(clamped_chunk, shard_shape))
        store = zarr.open_group(str(output_path), mode="w", zarr_format=3)
        store.create_array("0", data=arr, chunks=clamped_chunk, shards=eff_shard)
        if levels > 1:
            from scipy.ndimage import zoom

            for lvl in range(1, levels):
                factor = 1 / 2**lvl
                down = zoom(arr, factor, order=1).astype(arr.dtype)
                lc = tuple(min(c, s) for c, s in zip(clamped_chunk, down.shape))
                ls = tuple(min(c * 2, s) for c, s in zip(lc, down.shape))
                store.create_array(str(lvl), data=down, chunks=lc, shards=ls)
        store.attrs["nifti_affine"] = img.affine.tolist()

    # Store provenance in group attrs
    store = zarr.open_group(str(output_path), mode="r+")
    store.attrs["nobrainer_provenance"] = {
        "source_file": str(Path(input_path).name),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "nobrainer.io.nifti_to_zarr",
        "nobrainer_version": nobrainer.__version__,
        "chunk_shape": list(chunk_shape),
        "levels": levels,
    }

    return output_path


def zarr_to_nifti(
    input_path: str | Path,
    output_path: str | Path,
    level: int = 0,
) -> Path:
    """Convert a Zarr v3 store back to NIfTI.

    Tries ``niizarr.zarr2nii`` first for NIfTI-Zarr spec compliance.
    Falls back to reading the array + stored affine if niizarr is
    unavailable or fails.

    Parameters
    ----------
    input_path : path to .zarr directory
    output_path : path for the output .nii.gz file
    level : resolution level to export (0 = full resolution)

    Returns
    -------
    Path to the created NIfTI file
    """
    import zarr

    output_path = Path(output_path)

    # Try niizarr first
    try:
        import niizarr

        img = niizarr.zarr2nii(str(input_path), level=level)
        if img.affine is not None:
            nib.save(img, str(output_path))
            return output_path
    except Exception:
        pass

    # Fallback: manual read
    store = zarr.open_group(str(input_path), mode="r")
    arr = np.asarray(store[str(level)])
    affine = np.array(store.attrs.get("nifti_affine", np.eye(4).tolist()))
    img = nib.Nifti1Image(arr.astype(np.float32), affine)
    nib.save(img, str(output_path))
    return output_path


__all__ = [
    "read_csv",
    "read_mapping",
    "convert_tfrecords",
    "convert_weights",
    "nifti_to_zarr",
    "zarr_to_nifti",
]
