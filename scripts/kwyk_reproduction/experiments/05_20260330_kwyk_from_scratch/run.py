#!/usr/bin/env python
"""Train KWYKMeshNet from scratch: mc=True vs mc=False, 50-class vs binary."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import setup_logging

log = setup_logging(__name__)

WORK_DIR = Path(__file__).parent.parent.parent
EXP_DIR = Path(__file__).parent


def predict_volume(model, img_path, block_shape, mc=False):
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
            out = model(tensor, mc=mc)
            all_preds.append(out.argmax(dim=1, keepdim=True).float().cpu().numpy())
    block_preds = np.concatenate(all_preds, axis=0)
    return _stitch_blocks(block_preds, grid, block_shape, pad, orig_shape, 1)[0].astype(
        np.int32
    )


def per_class_dice(pred, gt, n_classes):
    dices = []
    for c in range(1, n_classes):
        p = pred == c
        g = gt == c
        inter = (p & g).sum()
        total = p.sum() + g.sum()
        dices.append(2.0 * inter / total if total > 0 else 1.0)
    return np.array(dices)


def binary_dice(pred, gt):
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return 2.0 * inter / total if total > 0 else 1.0


def train_kwyk(name, n_classes, label_mapping, mc_train, epochs=50):
    from nobrainer.models import get as get_model
    from nobrainer.processing.dataset import Dataset, _load_label_mapping
    from nobrainer.training import get_device

    log.info(
        "=== %s: n_classes=%d, mc_train=%s, epochs=%d ===",
        name,
        n_classes,
        mc_train,
        epochs,
    )
    block_shape = (32, 32, 32)
    device = get_device()

    # Load data
    train_pairs, val_pairs = [], []
    with open(WORK_DIR / "kwyk_sanity_manifest.csv") as f:
        for row in csv.DictReader(f):
            p = (row["t1w_path"], row["label_path"])
            if row["split"] == "train":
                train_pairs.append(p)
            elif row["split"] == "test":
                val_pairs.append(p)  # use test as val for sanity

    ds = (
        Dataset.from_files(train_pairs, block_shape=block_shape, n_classes=n_classes)
        .batch(32)
        .binarize(label_mapping)
    )

    # Remap function for eval
    remap_fn = None
    if label_mapping and label_mapping != "binary":
        remap_fn = _load_label_mapping(label_mapping)

    # Create model
    model = get_model("kwyk_meshnet")(
        n_classes=n_classes,
        filters=96,
        receptive_field=37,
        dropout_type="bernoulli",
        dropout_rate=0.25,
        sigma_init=0.0001,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Test volume
    img_path, lbl_path = val_pairs[0]
    gt_raw = np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
    if remap_fn:
        gt_arr = remap_fn(torch.from_numpy(gt_raw)).numpy()
    else:
        gt_arr = (gt_raw > 0).astype(np.int32)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in ds.dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                images = batch[0].to(device)
                labels = batch[1].to(device)
            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer.zero_grad()
            pred = model(images, mc=mc_train)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        msg = f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}"

        if (epoch + 1) % 10 == 0 or epoch == 0:
            pred_vol = predict_volume(model, img_path, block_shape, mc=False)
            if n_classes == 2:
                d = binary_dice(pred_vol, gt_arr)
                msg += f" dice_det={d:.4f}"
            else:
                cd = per_class_dice(pred_vol, gt_arr, n_classes)
                msg += f" dice_det={cd.mean():.4f}/{cd.max():.4f}"

        log.info(msg)

    elapsed = time.time() - t0
    log.info("  Completed in %.1fs", elapsed)

    # Final eval
    pred_vol = predict_volume(model, img_path, block_shape, mc=False)
    if n_classes == 2:
        d = binary_dice(pred_vol, gt_arr)
        log.info("  FINAL [det]: dice=%.4f", d)
    else:
        cd = per_class_dice(pred_vol, gt_arr, n_classes)
        log.info("  FINAL [det]: mean=%.4f, max=%.4f", cd.mean(), cd.max())

    if mc_train:
        pred_vol = predict_volume(model, img_path, block_shape, mc=True)
        if n_classes == 2:
            d = binary_dice(pred_vol, gt_arr)
            log.info("  FINAL [mc]: dice=%.4f", d)
        else:
            cd = per_class_dice(pred_vol, gt_arr, n_classes)
            log.info("  FINAL [mc]: mean=%.4f, max=%.4f", cd.mean(), cd.max())

    return model


def main():
    # A: 50-class, mc=True during training (current default)
    train_kwyk("A_50class_mcTrue", 50, "50-class", mc_train=True, epochs=50)

    # B: 50-class, mc=False during training (deterministic forward)
    train_kwyk("B_50class_mcFalse", 50, "50-class", mc_train=False, epochs=50)

    # C: Binary, mc=True during training
    train_kwyk("C_binary_mcTrue", 2, "binary", mc_train=True, epochs=50)

    # D: Binary, mc=False during training
    train_kwyk("D_binary_mcFalse", 2, "binary", mc_train=False, epochs=50)


if __name__ == "__main__":
    main()
