#!/usr/bin/env python
"""Build a manifest CSV from the original KWYK dataset (PAC brain volumes).

The KWYK dataset contains paired files:
  - pac_<ID>_orig.nii.gz  (T1w image)
  - pac_<ID>_aseg.nii.gz  (FreeSurfer aparc+aseg label)

Usage:
    python build_kwyk_manifest.py --data-dir ../data/SharedData/segmentation/freesurfer_asegs \
        --output-csv kwyk_manifest.csv --n-subjects 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description="Build manifest from KWYK PAC dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing pac_*_orig.nii.gz and pac_*_aseg.nii.gz files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="kwyk_manifest.csv",
        help="Output manifest CSV path",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Number of subjects to include (default: all)",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=int,
        default=[80, 10, 10],
        help="Train/val/test split percentages (default: 80 10 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and split (default: 42)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # Find all paired subjects
    orig_files = sorted(data_dir.glob("pac_*_orig.nii.gz"))
    pairs = []
    for orig in orig_files:
        # Extract subject ID: pac_<ID>_orig.nii.gz -> <ID>
        stem = orig.name  # pac_123_orig.nii.gz
        subj_id = stem.replace("pac_", "").replace("_orig.nii.gz", "")
        aseg = data_dir / f"pac_{subj_id}_aseg.nii.gz"
        if aseg.exists():
            pairs.append((subj_id, str(orig), str(aseg)))

    print(f"Found {len(pairs)} paired subjects in {data_dir}")

    if not pairs:
        raise SystemExit("No paired (orig, aseg) files found.")

    # Shuffle and subsample
    random.seed(args.seed)
    random.shuffle(pairs)
    if args.n_subjects is not None:
        pairs = pairs[: args.n_subjects]
        print(f"Subsampled to {len(pairs)} subjects")

    # Split
    n = len(pairs)
    train_pct, val_pct, test_pct = args.split
    assert train_pct + val_pct + test_pct == 100
    n_train = int(n * train_pct / 100)
    n_val = int(n * val_pct / 100)
    # rest goes to test

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)

    # Write manifest
    output_csv = Path(args.output_csv)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "dataset_id", "t1w_path", "label_path", "split"])
        for (subj_id, orig_path, aseg_path), split in zip(pairs, splits):
            writer.writerow([f"pac_{subj_id}", "kwyk", orig_path, aseg_path, split])

    # Summary
    from collections import Counter

    split_counts = Counter(splits)
    print(f"Manifest written to {output_csv}")
    print(
        f"  train: {split_counts['train']}, val: {split_counts['val']}, test: {split_counts['test']}"
    )


if __name__ == "__main__":
    main()
