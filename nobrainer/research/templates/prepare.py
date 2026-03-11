"""Standard data preparation script for autoresearch.

Usage
-----
    python prepare.py --data-dir /path/to/nifti --val-fraction 0.2

Writes ``data_manifest.json`` in the current directory listing
train/val split paths.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import click


@click.command()
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing NIfTI files (*.nii or *.nii.gz).",
)
@click.option(
    "--val-fraction",
    default=0.2,
    type=float,
    show_default=True,
    help="Fraction of data for validation.",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Random seed for train/val split.",
)
@click.option(
    "--output",
    default="data_manifest.json",
    show_default=True,
    help="Output manifest filename.",
)
def prepare(*, data_dir: str, val_fraction: float, seed: int, output: str) -> None:
    """Validate NIfTI dataset and write train/val split manifest."""
    data_path = Path(data_dir)
    niftis = sorted(list(data_path.glob("*.nii")) + list(data_path.glob("*.nii.gz")))
    if not niftis:
        raise click.ClickException(f"No NIfTI files found in {data_dir}")

    random.seed(seed)
    shuffled = list(niftis)
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    manifest = {
        "data_dir": str(data_path.resolve()),
        "n_total": len(shuffled),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "train": [str(p) for p in train_paths],
        "val": [str(p) for p in val_paths],
    }

    output_path = Path(output)
    output_path.write_text(json.dumps(manifest, indent=2))
    click.echo(
        f"Manifest written to {output_path}: "
        f"{len(train_paths)} train, {len(val_paths)} val"
    )


if __name__ == "__main__":
    prepare()
