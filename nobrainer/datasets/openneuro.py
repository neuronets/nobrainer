"""Fetch datasets from OpenNeuro and OpenNeuro Derivatives via DataLad.

Requires the ``[versioning]`` extra (``datalad >= 0.19``) and the
``git-annex`` PyPI package (``uv tool install git-annex`` or
``pip install git-annex``).

Examples
--------
Fetch fmriprep derivatives and get T1w + aparc+aseg pairs::

    from nobrainer.datasets.openneuro import (
        install_derivatives,
        find_subject_pairs,
        write_manifest,
    )

    ds_path = install_derivatives("ds000114", "/tmp/data")
    pairs = find_subject_pairs(ds_path)
    write_manifest(pairs, "manifest.csv")

Fetch a raw OpenNeuro dataset::

    from nobrainer.datasets.openneuro import install_dataset
    ds_path = install_dataset("ds000114", "/tmp/data")

Fetch specific files without auto-discovery::

    from nobrainer.datasets.openneuro import (
        install_derivatives,
        glob_dataset,
        fetch_files,
    )

    ds_path = install_derivatives("ds000114", "/tmp/data")
    bold_files = glob_dataset(ds_path, "sub-*/func/*_bold.nii.gz")
    fetched = fetch_files(ds_path, bold_files[:5])
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_OPENNEURO_GH = "https://github.com/OpenNeuroDatasets"
_OPENNEURO_DERIV_GH = "https://github.com/OpenNeuroDerivatives"


def _dl():
    """Lazy import of datalad.api."""
    from nobrainer.datasets import _check_datalad

    return _check_datalad()


# ---------------------------------------------------------------------------
# Install (lightweight clone, no bulk download)
# ---------------------------------------------------------------------------


def install_dataset(
    dataset_id: str,
    path: str | Path,
) -> Path:
    """Clone an OpenNeuro dataset (metadata only, no file content).

    Parameters
    ----------
    dataset_id : str
        OpenNeuro accession (e.g. ``"ds000114"``).
    path : str or Path
        Base directory.  The dataset is cloned into
        ``<path>/<dataset_id>``.

    Returns
    -------
    Path
        Absolute path to the installed dataset directory.
    """
    dl = _dl()
    dest = Path(path) / dataset_id
    if dest.exists():
        logger.info("Dataset %s already at %s", dataset_id, dest)
        return dest.resolve()

    source = f"{_OPENNEURO_GH}/{dataset_id}.git"
    logger.info("Installing %s from %s", dataset_id, source)
    dl.install(source=source, path=str(dest))
    return dest.resolve()


def install_derivatives(
    dataset_id: str,
    path: str | Path,
    derivative: str = "fmriprep",
) -> Path:
    """Clone an OpenNeuro Derivatives dataset (metadata only).

    Parameters
    ----------
    dataset_id : str
        OpenNeuro accession (e.g. ``"ds000114"``).
    path : str or Path
        Base directory.  Cloned into ``<path>/<dataset_id>-<derivative>``.
    derivative : str
        Pipeline name (default ``"fmriprep"``).  Common values:
        ``"fmriprep"``, ``"mriqc"``, ``"freesurfer"``.

    Returns
    -------
    Path
        Absolute path to the installed derivative directory.
    """
    dl = _dl()
    dest = Path(path) / f"{dataset_id}-{derivative}"
    if dest.exists():
        logger.info("Derivative %s-%s already at %s", dataset_id, derivative, dest)
        return dest.resolve()

    source = f"{_OPENNEURO_DERIV_GH}/{dataset_id}-{derivative}.git"
    logger.info("Installing %s-%s from %s", dataset_id, derivative, source)
    dl.install(source=source, path=str(dest))
    return dest.resolve()


# ---------------------------------------------------------------------------
# File discovery and download
# ---------------------------------------------------------------------------


def glob_dataset(
    dataset_dir: str | Path,
    pattern: str,
) -> list[Path]:
    """Glob a DataLad dataset directory (metadata only, no download).

    Works on the git tree — returned paths may be git-annex symlinks
    whose content hasn't been fetched yet.

    Parameters
    ----------
    dataset_dir : str or Path
        Root of the DataLad dataset.
    pattern : str
        Glob pattern (e.g. ``"sub-*/anat/*_T1w.nii.gz"``).

    Returns
    -------
    list of Path
        Sorted matching paths.
    """
    return sorted(Path(dataset_dir).glob(pattern))


def fetch_files(
    dataset_dir: str | Path,
    paths: list[str | Path],
) -> list[Path]:
    """Download specific files from a DataLad dataset.

    Parameters
    ----------
    dataset_dir : str or Path
        Root of the DataLad dataset.
    paths : list of str or Path
        Files to download (absolute or relative to *dataset_dir*).

    Returns
    -------
    list of Path
        Paths whose content was successfully downloaded.
    """
    dl = _dl()
    dataset_dir = Path(dataset_dir)

    try:
        dl.get([str(p) for p in paths], dataset=str(dataset_dir))
    except Exception as exc:
        logger.warning("datalad get failed: %s", exc)

    return [p for p in (Path(x) for x in paths) if _file_ok(p)]


# ---------------------------------------------------------------------------
# Paired file discovery (structural MRI)
# ---------------------------------------------------------------------------


def _extract_subject_id(path: Path) -> str:
    """Extract ``sub-XX`` from a BIDS-style path.

    Checks directory components first (``sub-01/anat/...``), then
    parses the filename (``sub-01_desc-preproc_T1w.nii.gz``).
    """
    # Check directory parts (e.g. .../sub-01/anat/...)
    for part in path.parts[:-1]:  # skip filename
        if part.startswith("sub-"):
            return part
    # Parse from filename
    name = path.name
    if name.startswith("sub-"):
        return name.split("_")[0]
    return name


def _file_ok(p: Path) -> bool:
    """True if *p* is a real file with nonzero size."""
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def find_subject_pairs(
    dataset_dir: str | Path,
    feature_pattern: str | None = None,
    label_pattern: str | None = None,
    native_space: bool = True,
    download: bool = True,
) -> list[dict[str, str]]:
    """Discover and optionally download paired (feature, label) files.

    The default patterns find native-space preprocessed T1w images and
    aparc+aseg parcellations from fmriprep derivatives.

    Strategy:

    1. Glob the dataset tree (git metadata only) to find label files.
    2. For each label, find the matching feature file for the same
       subject.
    3. Download each pair via ``datalad get``.
    4. Verify both files are accessible before including them.

    Parameters
    ----------
    dataset_dir : str or Path
        Root of a DataLad dataset (typically an fmriprep derivative).
    feature_pattern : str or None
        Glob for feature files.  When *None*, discovers the best
        native-space T1w pattern automatically.
    label_pattern : str or None
        Glob for label files.  When *None*, tries
        ``*desc-aparcaseg_dseg.nii.gz`` then ``*desc-aseg_dseg.nii.gz``.
    native_space : bool
        Prefer native-space files (no ``space-`` token).  Default True.
    download : bool
        If True (default), download each pair via ``datalad get``.

    Returns
    -------
    list of dict
        Each dict: ``{"subject_id", "t1w_path", "label_path"}``.
    """
    dataset_dir = Path(dataset_dir)
    pairs: list[dict[str, str]] = []

    # --- Discover label files ---
    if label_pattern is not None:
        label_files = glob_dataset(dataset_dir, label_pattern)
    else:
        label_files = []
        for pat in [
            "sub-*/anat/*desc-aparcaseg_dseg.nii.gz",
            "sub-*/anat/*desc-aseg_dseg.nii.gz",
        ]:
            label_files = glob_dataset(dataset_dir, pat)
            if label_files:
                logger.info("Found %d labels matching %s", len(label_files), pat)
                break

    if not label_files:
        logger.warning("No label files found in %s", dataset_dir)
        return pairs

    # --- Match each label to a feature file ---
    for label_path in label_files:
        sub_id = _extract_subject_id(label_path)
        anat_dir = label_path.parent

        if feature_pattern is not None:
            feat_candidates = sorted(anat_dir.glob(feature_pattern))
        else:
            feat_candidates = [
                p
                for p in anat_dir.glob(f"{sub_id}*desc-preproc_T1w.nii.gz")
                if (not native_space) or ("space-" not in p.name)
            ]
            if not feat_candidates:
                feat_candidates = sorted(anat_dir.glob(f"{sub_id}*_T1w.nii.gz"))[:1]

        if not feat_candidates:
            logger.warning("No feature file for %s", sub_id)
            continue

        feat_path = feat_candidates[0]

        if download:
            logger.info("Downloading pair for %s", sub_id)
            fetch_files(dataset_dir, [feat_path, label_path])

        feat_ok = _file_ok(feat_path) if download else True
        label_ok = _file_ok(label_path) if download else True

        if feat_ok and label_ok:
            pairs.append(
                {
                    "subject_id": sub_id,
                    "t1w_path": str(feat_path),
                    "label_path": str(label_path),
                }
            )
        else:
            logger.warning("Skipping %s: files not accessible", sub_id)

    logger.info("Found %d paired subjects in %s", len(pairs), dataset_dir.name)
    return pairs


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def write_manifest(
    pairs: list[dict[str, str]],
    output_path: str | Path,
    split_ratios: tuple[int, int, int] = (80, 10, 10),
    seed: int = 42,
) -> Path:
    """Write a manifest CSV with train/val/test split.

    Parameters
    ----------
    pairs : list of dict
        Each dict has ``"subject_id"``, ``"t1w_path"``, ``"label_path"``.
        Optionally ``"dataset_id"``.
    output_path : str or Path
        Destination CSV.
    split_ratios : tuple of int
        (train, val, test) percentages.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    Path
        Written CSV path.
    """
    import csv

    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))
    total = sum(split_ratios)
    n_train = int(len(pairs) * split_ratios[0] / total)
    n_val = int(len(pairs) * split_ratios[1] / total)

    for i, idx in enumerate(indices):
        if i < n_train:
            pairs[idx]["split"] = "train"
        elif i < n_train + n_val:
            pairs[idx]["split"] = "val"
        else:
            pairs[idx]["split"] = "test"

    fieldnames = ["subject_id", "dataset_id", "t1w_path", "label_path", "split"]
    if not any("dataset_id" in p for p in pairs):
        fieldnames = [f for f in fieldnames if f != "dataset_id"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)

    counts = {
        s: sum(1 for p in pairs if p.get("split") == s)
        for s in ("train", "val", "test")
    }
    logger.info(
        "Manifest: %s — %d subjects (train=%d, val=%d, test=%d)",
        output_path,
        len(pairs),
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return output_path
