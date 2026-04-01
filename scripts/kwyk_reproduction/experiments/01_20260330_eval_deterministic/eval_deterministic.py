#!/usr/bin/env python
"""Evaluate Bayesian models in deterministic mode (mc=False).

Quick diagnostic: do the weights contain useful information that MC noise destroys?
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import setup_logging

log = setup_logging(__name__)


def per_class_dice(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class Dice for classes 1..n_classes-1."""
    dice = np.zeros(n_classes - 1)
    for c in range(1, n_classes):
        p = (pred == c)
        g = (gt == c)
        inter = (p & g).sum()
        total = p.sum() + g.sum()
        dice[c - 1] = 2.0 * inter / total if total > 0 else 1.0
    return dice


def predict_volume(model, img_path, block_shape, mc=False):
    """Block-based prediction on a single volume."""
    from nobrainer.prediction import _pad_to_multiple, _extract_blocks, _stitch_blocks
    from nobrainer.training import get_device

    device = get_device()
    img = nib.load(str(img_path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    orig_shape = arr.shape[:3]
    padded, pad = _pad_to_multiple(arr, block_shape)
    blocks, grid = _extract_blocks(padded, block_shape)

    model = model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(blocks), 4):
            chunk = blocks[start:start + 4]
            tensor = torch.from_numpy(chunk[:, None]).to(device)
            out = model(tensor, mc=mc)
            labels = out.argmax(dim=1, keepdim=True).float()
            all_preds.append(labels.cpu().numpy())

    block_preds = np.concatenate(all_preds, axis=0)
    full = _stitch_blocks(block_preds, grid, block_shape, pad, orig_shape, 1)[0]
    return full.astype(np.int32)


def main():
    from nobrainer.processing.segmentation import Segmentation
    from nobrainer.processing.dataset import _load_label_mapping

    work_dir = Path(__file__).parent.parent.parent
    manifest_path = work_dir / "kwyk_manifest.csv"
    remap_fn = _load_label_mapping("50-class")
    n_classes = 50

    # Load test pairs
    pairs = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                pairs.append((row["t1w_path"], row["label_path"]))

    log.info("Test volumes: %d", len(pairs))

    # Evaluate each variant
    variants = [
        "kwyk_smoke_bwn_multi",
        "kwyk_smoke_bvwn_multi_prior",
        "kwyk_smoke_bayesian_gaussian",
    ]

    results = []
    for variant in variants:
        ckpt_dir = work_dir / "checkpoints" / variant
        if not (ckpt_dir / "model.pth").exists():
            log.warning("Skipping %s — no checkpoint", variant)
            continue

        log.info("=== %s ===", variant)
        seg = Segmentation.load(ckpt_dir)
        model = seg.model_
        block_shape = seg.block_shape_ or (32, 32, 32)

        for mc_mode in [False, True]:
            mode_name = "mc" if mc_mode else "deterministic"
            all_dice = []

            for idx, (img_path, lbl_path) in enumerate(pairs[:3]):  # first 3 for speed
                pred_arr = predict_volume(model, img_path, block_shape, mc=mc_mode)
                gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
                gt_arr = remap_fn(torch.from_numpy(gt_arr)).numpy()

                cd = per_class_dice(pred_arr, gt_arr, n_classes)
                avg = float(cd.mean())
                all_dice.append(avg)
                log.info("  [%s] vol %d: avg_dice=%.4f max=%.4f", mode_name, idx + 1, avg, cd.max())

            mean_dice = float(np.mean(all_dice))
            results.append({"variant": variant, "mode": mode_name, "mean_dice": mean_dice})
            log.info("  [%s] MEAN: %.4f", mode_name, mean_dice)

    # Save results
    out_path = Path(__file__).parent / "results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "mode", "mean_dice"])
        w.writeheader()
        w.writerows(results)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
