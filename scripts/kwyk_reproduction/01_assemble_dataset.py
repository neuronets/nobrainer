#!/usr/bin/env python
"""Assemble training dataset from OpenNeuro fmriprep derivatives via DataLad.

Usage:
    python 01_assemble_dataset.py --datasets ds000114 --output-csv manifest.csv
    python 01_assemble_dataset.py --datasets ds000114 ds000228 ds002609 \
        --output-csv manifest.csv --label-mapping binary --split 80 10 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def install_dataset(dataset_id: str, base_dir: Path) -> Path:
    """Install an OpenNeuro fmriprep derivative via DataLad Python API."""
    import datalad.api as dl

    repo_url = f"https://github.com/OpenNeuroDerivatives/{dataset_id}-fmriprep.git"
    dest = base_dir / f"{dataset_id}-fmriprep"

    if dest.exists():
        log.info("Dataset %s already installed at %s", dataset_id, dest)
        return dest

    log.info("Installing %s from %s", dataset_id, repo_url)
    dl.install(source=repo_url, path=str(dest))
    return dest


def datalad_get(paths: list[str | Path], dataset_dir: str | Path) -> None:
    """Download file content via DataLad Python API."""
    import datalad.api as dl

    try:
        dl.get([str(p) for p in paths], dataset=str(dataset_dir))
    except Exception as exc:
        log.warning("datalad get failed: %s", exc)


def _file_accessible(p: Path) -> bool:
    """Check that *p* is a real file with content (not a broken annex symlink)."""
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def _extract_subject_id(path: Path) -> str:
    """Extract the subject ID (e.g. 'sub-01') from a BIDS-style path."""
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return path.stem.split("_")[0]


def find_subject_pairs(dataset_dir: Path) -> list[dict[str, str]]:
    """Find T1w + aparc+aseg pairs, discover patterns, and download per subject.

    Strategy:
    1. Inspect the dataset tree (git metadata only, no content yet) to
       find which aparc+aseg pattern exists.
    2. For each subject with an aparc+aseg, find the matching native-space
       T1w (same subject, no ``space-`` in filename).
    3. Download each (T1w, label) pair via ``datalad get``.
    """
    pairs: list[dict[str, str]] = []

    # --- Discover label pattern -------------------------------------------
    label_patterns = [
        "sub-*/anat/*desc-aparcaseg_dseg.nii.gz",
        "sub-*/anat/*desc-aseg_dseg.nii.gz",
    ]
    label_files: list[Path] = []
    label_pat_used = ""
    for pat in label_patterns:
        label_files = sorted(dataset_dir.glob(pat))
        if label_files:
            label_pat_used = pat
            log.info("Found %d label files matching %s", len(label_files), pat)
            break

    if not label_files:
        log.warning("No aparc+aseg / aseg labels found in %s", dataset_dir)
        return pairs

    # --- For each subject, find its native-space T1w ----------------------
    for label_path in label_files:
        sub_id = _extract_subject_id(label_path)
        anat_dir = label_path.parent

        # Native-space T1w: no 'space-' token in the filename
        t1w_candidates = [
            p
            for p in anat_dir.glob(f"{sub_id}*desc-preproc_T1w.nii.gz")
            if "space-" not in p.name
        ]
        if not t1w_candidates:
            # Fall back to any T1w for this subject
            t1w_candidates = sorted(anat_dir.glob(f"{sub_id}*_T1w.nii.gz"))[:1]

        if not t1w_candidates:
            log.warning("No T1w found for %s in %s", sub_id, anat_dir)
            continue

        t1w_path = t1w_candidates[0]

        # Download this pair
        log.info("Downloading pair for %s", sub_id)
        datalad_get([t1w_path, label_path], dataset_dir)

        # Verify content is accessible
        if _file_accessible(t1w_path) and _file_accessible(label_path):
            pairs.append(
                {
                    "subject_id": sub_id,
                    "t1w_path": str(t1w_path),
                    "label_path": str(label_path),
                }
            )
        else:
            log.warning("Skipping %s: files not accessible after datalad get", sub_id)

    log.info(
        "Found %d subject pairs in %s (label pattern: %s)",
        len(pairs),
        dataset_dir.name,
        label_pat_used,
    )
    return pairs


def conform_volume(path: str, target_shape: tuple = (256, 256, 256)) -> bool:
    """Check if volume is conformable; resample if needed."""
    img = nib.load(path)
    if img.shape[:3] == target_shape:
        return True

    log.info(
        "Resampling %s from %s to %s", Path(path).name, img.shape[:3], target_shape
    )
    from nibabel.processing import conform

    conformed = conform(img, out_shape=target_shape, voxel_size=(1.0, 1.0, 1.0))
    nib.save(conformed, path)
    return True


def split_manifest(
    rows: list[dict],
    ratios: tuple[int, int, int] = (80, 10, 10),
) -> list[dict]:
    """Add 'split' column with stratified train/val/test split."""
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(rows))
    total = sum(ratios)
    n_train = int(len(rows) * ratios[0] / total)
    n_val = int(len(rows) * ratios[1] / total)

    for i, idx in enumerate(indices):
        if i < n_train:
            rows[idx]["split"] = "train"
        elif i < n_train + n_val:
            rows[idx]["split"] = "val"
        else:
            rows[idx]["split"] = "test"

    return rows


def main():
    parser = argparse.ArgumentParser(description="Assemble dataset from OpenNeuro")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ds000114"],
        help="OpenNeuro dataset IDs",
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--output-csv", default="manifest.csv", help="Output manifest CSV"
    )
    parser.add_argument(
        "--label-mapping",
        default="binary",
        help="Label mapping: binary, 6-class, 50-class, 115-class",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=int,
        default=[80, 10, 10],
        help="Train/val/test split percentages",
    )
    parser.add_argument("--conform", action="store_true", help="Resample to 256³ @ 1mm")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for ds_id in args.datasets:
        ds_dir = install_dataset(ds_id, output_dir)
        pairs = find_subject_pairs(ds_dir)
        for p in pairs:
            p["dataset_id"] = ds_id
        all_pairs.extend(pairs)

    if not all_pairs:
        log.error("No subject pairs found. Check dataset IDs and network access.")
        sys.exit(1)

    # Optionally conform volumes
    if args.conform:
        for row in all_pairs:
            conform_volume(row["t1w_path"])

    # Split
    all_pairs = split_manifest(all_pairs, tuple(args.split))

    # Write manifest
    csv_path = Path(args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "dataset_id", "t1w_path", "label_path", "split"],
        )
        writer.writeheader()
        writer.writerows(all_pairs)

    n_train = sum(1 for r in all_pairs if r["split"] == "train")
    n_val = sum(1 for r in all_pairs if r["split"] == "val")
    n_test = sum(1 for r in all_pairs if r["split"] == "test")

    log.info(
        "Manifest written to %s: %d subjects (train=%d, val=%d, test=%d)",
        csv_path,
        len(all_pairs),
        n_train,
        n_val,
        n_test,
    )


if __name__ == "__main__":
    main()
