#!/usr/bin/env python
"""Assemble training dataset from OpenNeuro fmriprep derivatives.

Uses :mod:`nobrainer.datasets.openneuro` to install datasets via DataLad
and discover paired (T1w, aparc+aseg) files per subject.

Usage:
    python 01_assemble_dataset.py --datasets ds000114 --output-csv manifest.csv
    python 01_assemble_dataset.py --datasets ds000114 ds000228 ds002609 \
        --output-csv manifest.csv --label-mapping binary --split 80 10 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


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

    from nobrainer.datasets.openneuro import (
        find_subject_pairs,
        install_derivatives,
        write_manifest,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for ds_id in args.datasets:
        ds_dir = install_derivatives(ds_id, output_dir)
        pairs = find_subject_pairs(ds_dir)
        for p in pairs:
            p["dataset_id"] = ds_id
        all_pairs.extend(pairs)

    if not all_pairs:
        log.error("No subject pairs found. Check dataset IDs and network access.")
        raise SystemExit(1)

    # Optionally conform volumes
    if args.conform:
        import nibabel as nib
        from nibabel.processing import conform

        for row in all_pairs:
            img = nib.load(row["t1w_path"])
            if img.shape[:3] != (256, 256, 256):
                log.info("Conforming %s", Path(row["t1w_path"]).name)
                conformed = conform(
                    img, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0)
                )
                nib.save(conformed, row["t1w_path"])

    write_manifest(all_pairs, args.output_csv, tuple(args.split))


if __name__ == "__main__":
    main()
