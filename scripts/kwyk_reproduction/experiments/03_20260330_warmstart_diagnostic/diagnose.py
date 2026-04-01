#!/usr/bin/env python
"""Diagnose warm-start transfer from MeshNet to KWYKMeshNet."""

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

WORK_DIR = Path(__file__).parent.parent.parent
EXP_DIR = Path(__file__).parent


def predict_volume_simple(model, img_path, block_shape, mc=False):
    """Block-based prediction."""
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
            try:
                out = model(tensor, mc=mc)
            except TypeError:
                out = model(tensor)
            labels = out.argmax(dim=1, keepdim=True).float()
            all_preds.append(labels.cpu().numpy())

    block_preds = np.concatenate(all_preds, axis=0)
    full = _stitch_blocks(block_preds, grid, block_shape, pad, orig_shape, 1)[0]
    return full.astype(np.int32)


def main():
    from nobrainer.models import get as get_model
    from nobrainer.processing.segmentation import Segmentation
    from nobrainer.processing.dataset import _load_label_mapping

    remap_fn = _load_label_mapping("50-class")
    n_classes = 50
    block_shape = (32, 32, 32)

    # Load test pairs
    pairs = []
    with open(WORK_DIR / "kwyk_manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                pairs.append((row["t1w_path"], row["label_path"]))

    # Use first test volume
    img_path, lbl_path = pairs[0]
    gt_arr = remap_fn(torch.from_numpy(
        np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
    )).numpy()

    # ---- Step 1: Load trained MeshNet and eval ----
    log.info("=== Step 1: Evaluate trained MeshNet ===")
    seg = Segmentation.load(WORK_DIR / "checkpoints" / "kwyk_smoke_meshnet")
    det_model = seg.model_
    log.info("MeshNet type: %s", type(det_model).__name__)
    log.info("MeshNet params: %d", sum(p.numel() for p in det_model.parameters()))

    pred = predict_volume_simple(det_model, img_path, block_shape, mc=False)
    # per-class dice
    dices = []
    for c in range(1, n_classes):
        p = (pred == c); g = (gt_arr == c)
        inter = (p & g).sum(); total = p.sum() + g.sum()
        dices.append(2.0 * inter / total if total > 0 else 1.0)
    log.info("MeshNet Dice: mean=%.4f, max=%.4f", np.mean(dices), np.max(dices))

    # ---- Step 2: Create KWYKMeshNet and warm-start ----
    log.info("=== Step 2: Warm-start KWYKMeshNet from MeshNet ===")
    model_args = {
        "n_classes": n_classes,
        "filters": 96,
        "receptive_field": 37,
        "dropout_type": "bernoulli",
        "dropout_rate": 0.25,
        "sigma_init": 0.0001,
    }

    kwyk_factory = get_model("kwyk_meshnet")
    kwyk_model = kwyk_factory(**model_args)
    log.info("KWYKMeshNet type: %s", type(kwyk_model).__name__)
    log.info("KWYKMeshNet params: %d", sum(p.numel() for p in kwyk_model.parameters()))

    # Print layer comparison
    log.info("--- MeshNet layers ---")
    for name, param in det_model.named_parameters():
        log.info("  %s: %s", name, param.shape)

    log.info("--- KWYKMeshNet layers ---")
    for name, param in kwyk_model.named_parameters():
        log.info("  %s: %s", name, param.shape)

    # Run warm-start
    from nobrainer.models.bayesian.warmstart import warmstart_kwyk_from_deterministic

    meshnet_ckpt = WORK_DIR / "checkpoints" / "kwyk_smoke_meshnet" / "model.pth"
    n_transferred = warmstart_kwyk_from_deterministic(kwyk_model, str(meshnet_ckpt))
    log.info("Transferred %d layers", n_transferred)

    # ---- Step 3: Eval KWYKMeshNet BEFORE any Bayesian training ----
    log.info("=== Step 3: Evaluate warm-started KWYKMeshNet (mc=False) ===")
    pred = predict_volume_simple(kwyk_model, img_path, block_shape, mc=False)
    dices = []
    for c in range(1, n_classes):
        p = (pred == c); g = (gt_arr == c)
        inter = (p & g).sum(); total = p.sum() + g.sum()
        dices.append(2.0 * inter / total if total > 0 else 1.0)
    log.info("KWYKMeshNet (warm-start, mc=False) Dice: mean=%.4f, max=%.4f", np.mean(dices), np.max(dices))

    # Also test mc=True
    log.info("=== Step 4: Evaluate warm-started KWYKMeshNet (mc=True) ===")
    pred = predict_volume_simple(kwyk_model, img_path, block_shape, mc=True)
    dices = []
    for c in range(1, n_classes):
        p = (pred == c); g = (gt_arr == c)
        inter = (p & g).sum(); total = p.sum() + g.sum()
        dices.append(2.0 * inter / total if total > 0 else 1.0)
    log.info("KWYKMeshNet (warm-start, mc=True) Dice: mean=%.4f, max=%.4f", np.mean(dices), np.max(dices))

    # ---- Step 5: Check what 03_train_bayesian does ----
    log.info("=== Step 5: Check how training script loads warm-start ===")
    # Read the training script to see if it uses warmstart_kwyk_from_deterministic
    train_script = WORK_DIR / "03_train_bayesian.py"
    with open(train_script) as f:
        content = f.read()
    if "warmstart_kwyk_from_deterministic" in content:
        log.info("Training script uses warmstart_kwyk_from_deterministic")
    elif "warmstart_bayesian_from_deterministic" in content:
        log.info("Training script uses warmstart_bayesian_from_deterministic (WRONG for KWYK!)")
    else:
        log.info("No warmstart function found in training script — check manually")

    # Grep for the relevant line
    for line_no, line in enumerate(content.split("\n"), 1):
        if "warmstart" in line.lower() and not line.strip().startswith("#"):
            log.info("  Line %d: %s", line_no, line.strip())


if __name__ == "__main__":
    main()
