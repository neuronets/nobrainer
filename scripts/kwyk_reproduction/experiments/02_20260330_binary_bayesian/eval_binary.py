#!/usr/bin/env python
"""Evaluate binary Bayesian model in both mc=True and mc=False modes."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import compute_dice, setup_logging

log = setup_logging(__name__)

EXP_DIR = Path(__file__).parent
WORK_DIR = EXP_DIR.parent.parent


def predict_volume(model, img_path, block_shape, mc=False):
    """Block-based prediction with mc control."""
    from nobrainer.prediction import _extract_blocks, _pad_to_multiple, _stitch_blocks
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
            chunk = blocks[start : start + 4]
            tensor = torch.from_numpy(chunk[:, None]).to(device)
            if hasattr(model, "forward") and "mc" in model.forward.__code__.co_varnames:
                out = model(tensor, mc=mc)
            else:
                out = model(tensor)
            labels = out.argmax(dim=1, keepdim=True).float()
            all_preds.append(labels.cpu().numpy())

    block_preds = np.concatenate(all_preds, axis=0)
    full = _stitch_blocks(block_preds, grid, block_shape, pad, orig_shape, 1)[0]
    return (full > 0).astype(np.float32)


def main():
    from nobrainer.processing.segmentation import Segmentation

    manifest_path = WORK_DIR / "kwyk_sanity_manifest.csv"

    # Load test pairs
    pairs = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                pairs.append((row["t1w_path"], row["label_path"]))

    log.info("Test volumes: %d", len(pairs))

    results = []
    for variant in ["meshnet", "bwn_multi"]:
        ckpt_dir = EXP_DIR / "checkpoints" / variant
        if not (ckpt_dir / "model.pth").exists():
            log.warning("Skipping %s — no checkpoint", variant)
            continue

        seg = Segmentation.load(ckpt_dir)
        model = seg.model_
        block_shape = seg.block_shape_ or (32, 32, 32)

        mc_modes = [False] if variant == "meshnet" else [False, True]
        for mc_mode in mc_modes:
            mode_name = "mc" if mc_mode else "deterministic"
            dices = []

            for idx, (img_path, lbl_path) in enumerate(pairs):
                pred = predict_volume(model, img_path, block_shape, mc=mc_mode)
                gt = (
                    np.asarray(nib.load(lbl_path).dataobj, dtype=np.float32) > 0
                ).astype(np.float32)
                dice = compute_dice(pred, gt)
                dices.append(dice)
                log.info(
                    "  [%s/%s] vol %d: Dice=%.4f", variant, mode_name, idx + 1, dice
                )

            mean_d = float(np.mean(dices))
            results.append(
                {"variant": variant, "mode": mode_name, "mean_dice": f"{mean_d:.4f}"}
            )
            log.info("  [%s/%s] MEAN DICE: %.4f", variant, mode_name, mean_d)

    out_path = EXP_DIR / "results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "mode", "mean_dice"])
        w.writeheader()
        w.writerows(results)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
