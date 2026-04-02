#!/usr/bin/env python
"""Convert one shard of NIfTI volumes to a pre-created Zarr3 store.

Usage:
    python convert_zarr_shard.py --manifest manifest.csv --zarr-store data/store.zarr \
        --shard-idx 0 --subjects-per-shard 50

Called by SLURM job array — each task writes one shard independently.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--zarr-store", required=True)
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--subjects-per-shard", type=int, default=50)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the store (only shard 0 should do this)",
    )
    args = parser.parse_args()

    # Read manifest
    pairs = []
    subject_ids = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            pairs.append((row["t1w_path"], row["label_path"]))
            subject_ids.append(row["subject_id"])

    n_subjects = len(pairs)
    sps = args.subjects_per_shard
    start = args.shard_idx * sps
    end = min(start + sps, n_subjects)

    if start >= n_subjects:
        print(f"Shard {args.shard_idx}: no subjects (start={start} >= {n_subjects})")
        return

    store_path = Path(args.zarr_store)

    if args.create:
        # Create the store and arrays (only one task does this)
        D, H, W = 256, 256, 256
        n_shards = (n_subjects + sps - 1) // sps
        store = zarr.open_group(str(store_path), mode="w")
        store.create_array(
            "images",
            shape=(n_subjects, D, H, W),
            chunks=(1, 32, 32, 32),
            shards=(sps, D, H, W),
            dtype=np.float32,
        )
        store.create_array(
            "labels",
            shape=(n_subjects, D, H, W),
            chunks=(1, 32, 32, 32),
            shards=(sps, D, H, W),
            dtype=np.int32,
        )
        store.attrs["n_subjects"] = n_subjects
        store.attrs["subject_ids"] = subject_ids
        store.attrs["volume_shape"] = [D, H, W]
        print(f"Created store: {store_path} ({n_subjects} subjects, {n_shards} shards)")

        # Write partition JSON
        import json

        partitions = {"train": [], "val": [], "test": []}
        with open(args.manifest) as f:
            for row in csv.DictReader(f):
                partitions[row["split"]].append(row["subject_id"])
        part_path = str(store_path) + "_partition.json"
        with open(part_path, "w") as f:
            json.dump({"partitions": partitions}, f, indent=2)
        for k, v in partitions.items():
            print(f"  {k}: {len(v)} subjects")
    else:
        # Open existing store in append mode
        store = zarr.open_group(str(store_path), mode="r+")

    images_arr = store["images"]
    labels_arr = store["labels"]

    t0 = time.time()
    for i in range(start, end):
        img_path, lbl_path = pairs[i]
        # PAC data is already 256³ @ 1mm uint8/int32 — no conform needed
        img_data = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        lbl_data = np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
        images_arr[i] = img_data[:256, :256, :256]
        labels_arr[i] = lbl_data[:256, :256, :256]

        if (i - start + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i - start + 1) / elapsed
            print(
                f"  Shard {args.shard_idx}: {i - start + 1}/{end - start} "
                f"({rate:.1f} vol/s, {elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"Shard {args.shard_idx}: wrote {end - start} volumes in {elapsed:.1f}s "
        f"({(end - start) / elapsed:.1f} vol/s)"
    )


if __name__ == "__main__":
    main()
