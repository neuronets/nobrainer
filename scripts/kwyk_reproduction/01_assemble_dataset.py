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
import subprocess
import sys

import nibabel as nib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def install_dataset(dataset_id: str, base_dir: Path) -> Path:
    """Install an OpenNeuro fmriprep derivative via DataLad."""
    repo_url = f"https://github.com/OpenNeuroDerivatives/{dataset_id}-fmriprep.git"
    dest = base_dir / f"{dataset_id}-fmriprep"

    if dest.exists():
        log.info("Dataset %s already installed at %s", dataset_id, dest)
        return dest

    log.info("Installing %s from %s", dataset_id, repo_url)
    subprocess.run(
        ["datalad", "install", "--source", repo_url, str(dest)],
        check=True,
    )
    return dest


def get_files(dataset_dir: Path, pattern: str) -> list[Path]:
    """Get specific files via datalad get with glob pattern."""
    log.info("Getting files matching %s in %s", pattern, dataset_dir.name)
    result = subprocess.run(
        ["datalad", "get", "-d", str(dataset_dir), pattern],
        capture_output=True,
        text=True,
        cwd=str(dataset_dir),
    )
    if result.returncode != 0:
        log.warning(
            "datalad get returned %d: %s", result.returncode, result.stderr[:200]
        )

    return sorted(dataset_dir.glob(pattern))


def find_subject_pairs(
    dataset_dir: Path,
) -> list[dict[str, str]]:
    """Find T1w + aparc+aseg pairs for all subjects."""
    pairs = []

    # Try common fmriprep output patterns
    t1w_patterns = [
        "sub-*/anat/*desc-preproc_T1w.nii.gz",
        "sub-*/anat/*space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz",
        "sub-*/anat/*_T1w.nii.gz",
    ]
    label_patterns = [
        "sub-*/anat/*desc-aparcaseg_dseg.nii.gz",
        "sub-*/anat/*desc-aseg_dseg.nii.gz",
    ]

    t1w_files = []
    for pat in t1w_patterns:
        t1w_files = get_files(dataset_dir, pat)
        if t1w_files:
            break

    label_files = []
    for pat in label_patterns:
        label_files = get_files(dataset_dir, pat)
        if label_files:
            break

    if not t1w_files:
        log.warning("No T1w files found in %s", dataset_dir)
        return pairs

    # Match by subject ID
    t1w_by_sub = {}
    for f in t1w_files:
        sub = f.parts[-3] if "sub-" in f.parts[-3] else f.stem.split("_")[0]
        t1w_by_sub[sub] = f

    label_by_sub = {}
    for f in label_files:
        sub = f.parts[-3] if "sub-" in f.parts[-3] else f.stem.split("_")[0]
        label_by_sub[sub] = f

    for sub in sorted(set(t1w_by_sub) & set(label_by_sub)):
        pairs.append(
            {
                "subject_id": sub,
                "t1w_path": str(t1w_by_sub[sub]),
                "label_path": str(label_by_sub[sub]),
            }
        )

    log.info("Found %d subject pairs in %s", len(pairs), dataset_dir.name)
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
