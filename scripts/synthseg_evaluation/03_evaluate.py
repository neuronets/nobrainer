#!/usr/bin/env python
"""Evaluate a trained model with per-class Dice scoring.

Reuses the evaluation logic from the kwyk reproduction pipeline.

Usage:
    python 03_evaluate.py --model checkpoints/unet_real --manifest manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def per_class_dice(pred, gt, n_classes):
    """Compute Dice per class c=1..n_classes-1 (skip background)."""
    dice = np.zeros(n_classes - 1)
    for c in range(1, n_classes):
        p = (pred == c).astype(np.float64)
        g = (gt == c).astype(np.float64)
        intersection = (p * g).sum()
        total = p.sum() + g.sum()
        dice[c - 1] = 2.0 * intersection / total if total > 0 else 1.0
    return dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_classes = config["data"]["n_classes"]
    block_shape = tuple(config["data"]["block_shape"])
    label_mapping = config["data"]["label_mapping"]

    output_dir = Path(args.output_dir or args.model) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from nobrainer.processing.segmentation import Segmentation

    seg = Segmentation.load(args.model)

    # Load remap function
    remap_fn = None
    if label_mapping and label_mapping != "binary":
        from nobrainer.processing.dataset import _load_label_mapping

        remap_fn = _load_label_mapping(label_mapping)

    # Load test pairs
    pairs = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            if row["split"] == args.split:
                pairs.append((row["t1w_path"], row["label_path"]))

    log.info("Evaluating %d volumes", len(pairs))

    results = []
    all_dice = []
    for i, (img_path, lbl_path) in enumerate(pairs):
        gt = np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
        if remap_fn is not None:
            gt = remap_fn(gt)

        pred_img = seg.predict(img_path, block_shape=block_shape)
        pred = np.asarray(pred_img.dataobj, dtype=np.int32)

        dice = per_class_dice(pred, gt, n_classes)
        avg = float(dice.mean())
        all_dice.append(dice)
        results.append({"volume": Path(img_path).stem, "avg_dice": avg})
        log.info("  %d/%d: %s — Dice=%.4f", i + 1, len(pairs), Path(img_path).stem, avg)

    # Save results
    csv_path = output_dir / "dice_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["volume", "avg_dice"])
        w.writeheader()
        w.writerows(results)

    np.save(output_dir / "per_class_dice.npy", np.array(all_dice))

    avg_dices = [r["avg_dice"] for r in results]
    log.info("Class Dice: %.4f ± %.4f", np.mean(avg_dices), np.std(avg_dices))


if __name__ == "__main__":
    main()
