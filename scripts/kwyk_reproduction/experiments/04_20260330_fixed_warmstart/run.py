#!/usr/bin/env python
"""Fix warm-start, verify transfer, train Bayesian, evaluate."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import setup_logging  # noqa: E402

log = setup_logging(__name__)

WORK_DIR = Path(__file__).parent.parent.parent
EXP_DIR = Path(__file__).parent


def fixed_warmstart_kwyk(kwyk_model, det_weights_path):
    """Fixed warm-start: filter classifier, match encoder layers correctly."""
    from nobrainer.models.bayesian.vwn_layers import FFGConv3d

    state = torch.load(det_weights_path, weights_only=True)

    # Separate encoder convs from classifier
    encoder_convs = []
    classifier_weight = None
    classifier_bias = None
    for k in sorted(state.keys()):
        v = state[k]
        if k == "classifier.weight" and v.ndim == 5:
            classifier_weight = v
        elif k == "classifier.bias":
            classifier_bias = v
        elif "weight" in k and v.ndim == 5:
            encoder_convs.append((k, v))

    log.info("Found %d encoder convs + classifier in MeshNet", len(encoder_convs))

    # Transfer encoder convs to FFGConv3d layers
    kwyk_convs = [
        (n, m) for n, m in kwyk_model.named_modules() if isinstance(m, FFGConv3d)
    ]

    transferred = 0
    for (det_name, det_w), (kwyk_name, kwyk_conv) in zip(encoder_convs, kwyk_convs):
        if det_w.shape != kwyk_conv.v.shape:
            log.warning(
                "Shape mismatch: %s %s vs %s.v %s",
                det_name,
                det_w.shape,
                kwyk_name,
                kwyk_conv.v.shape,
            )
            continue
        kwyk_conv.v.data.copy_(det_w)
        norms = det_w.flatten(1).norm(dim=1).view_as(kwyk_conv.g)
        kwyk_conv.g.data.copy_(norms)
        transferred += 1
        log.info("  %s -> %s", det_name, kwyk_name)

    # Transfer classifier
    if classifier_weight is not None and hasattr(kwyk_model, "classifier"):
        kwyk_model.classifier.weight.data.copy_(classifier_weight)
        if classifier_bias is not None:
            kwyk_model.classifier.bias.data.copy_(classifier_bias)
        log.info("  classifier transferred")
        transferred += 1

    log.info("Total transferred: %d layers", transferred)
    return transferred


def predict_volume(model, img_path, block_shape, mc=False):
    """Block-based prediction."""
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
            try:
                out = model(tensor, mc=mc)
            except TypeError:
                out = model(tensor)
            labels = out.argmax(dim=1, keepdim=True).float()
            all_preds.append(labels.cpu().numpy())

    block_preds = np.concatenate(all_preds, axis=0)
    full = _stitch_blocks(block_preds, grid, block_shape, pad, orig_shape, 1)[0]
    return full.astype(np.int32)


def per_class_dice(pred, gt, n_classes):
    dices = []
    for c in range(1, n_classes):
        p = pred == c
        g = gt == c
        inter = (p & g).sum()
        total = p.sum() + g.sum()
        dices.append(2.0 * inter / total if total > 0 else 1.0)
    return np.array(dices)


def main():
    from nobrainer.models import get as get_model
    from nobrainer.processing.dataset import Dataset, _load_label_mapping
    from nobrainer.processing.segmentation import Segmentation

    n_classes = 50
    block_shape = (32, 32, 32)
    remap_fn = _load_label_mapping("50-class")

    # Test volume
    pairs = []
    with open(WORK_DIR / "kwyk_sanity_manifest.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                pairs.append((row["t1w_path"], row["label_path"]))
    img_path, lbl_path = pairs[0]
    gt_arr = remap_fn(
        torch.from_numpy(np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32))
    ).numpy()

    # ---- Step 1: Verify MeshNet baseline ----
    log.info("=== Step 1: MeshNet baseline ===")
    seg = Segmentation.load(WORK_DIR / "checkpoints" / "sanity_meshnet")
    det_model = seg.model_
    pred = predict_volume(det_model, img_path, block_shape)
    cd = per_class_dice(pred, gt_arr, n_classes)
    log.info("MeshNet: mean=%.4f, max=%.4f", cd.mean(), cd.max())

    # ---- Step 2: Fixed warm-start ----
    log.info("=== Step 2: Fixed warm-start ===")
    kwyk_factory = get_model("kwyk_meshnet")
    kwyk_model = kwyk_factory(
        n_classes=n_classes,
        filters=96,
        receptive_field=37,
        dropout_type="bernoulli",
        dropout_rate=0.25,
        sigma_init=0.0001,
    )
    meshnet_ckpt = WORK_DIR / "checkpoints" / "sanity_meshnet" / "model.pth"
    fixed_warmstart_kwyk(kwyk_model, meshnet_ckpt)

    # Eval immediately
    pred = predict_volume(kwyk_model, img_path, block_shape, mc=False)
    cd = per_class_dice(pred, gt_arr, n_classes)
    log.info(
        "KWYKMeshNet fixed warm-start (mc=False): mean=%.4f, max=%.4f",
        cd.mean(),
        cd.max(),
    )

    pred = predict_volume(kwyk_model, img_path, block_shape, mc=True)
    cd = per_class_dice(pred, gt_arr, n_classes)
    log.info(
        "KWYKMeshNet fixed warm-start (mc=True): mean=%.4f, max=%.4f",
        cd.mean(),
        cd.max(),
    )

    # ---- Step 3: Train Bayesian (5 subjects, 20 epochs) ----
    log.info("=== Step 3: Train Bayesian with fixed warm-start ===")
    manifest = WORK_DIR / "kwyk_sanity_manifest.csv"
    label_mapping = "50-class"

    train_pairs = []
    val_pairs = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            p = (row["t1w_path"], row["label_path"])
            if row["split"] == "train":
                train_pairs.append(p)
            elif row["split"] == "val":
                val_pairs.append(p)

    ds_train = (
        Dataset.from_files(train_pairs, block_shape=block_shape, n_classes=n_classes)
        .batch(32)
        .binarize(label_mapping)
    )

    from nobrainer.training import get_device

    device = get_device()
    kwyk_model = kwyk_model.to(device)
    optimizer = torch.optim.Adam(kwyk_model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        kwyk_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in ds_train.dataloader:
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
            pred_t = kwyk_model(images, mc=True)
            loss = criterion(pred_t, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        msg = f"Epoch {epoch + 1}/20: loss={avg_loss:.4f}"

        # Eval every 5 epochs
        if (epoch + 1) % 5 == 0:
            pred = predict_volume(kwyk_model, img_path, block_shape, mc=False)
            cd = per_class_dice(pred, gt_arr, n_classes)
            msg += f" dice_det={cd.mean():.4f}/{cd.max():.4f}"

            pred = predict_volume(kwyk_model, img_path, block_shape, mc=True)
            cd_mc = per_class_dice(pred, gt_arr, n_classes)
            msg += f" dice_mc={cd_mc.mean():.4f}/{cd_mc.max():.4f}"

        log.info(msg)

    # ---- Final eval ----
    log.info("=== Final evaluation ===")
    for mc_mode in [False, True]:
        pred = predict_volume(kwyk_model, img_path, block_shape, mc=mc_mode)
        cd = per_class_dice(pred, gt_arr, n_classes)
        mode = "mc" if mc_mode else "det"
        log.info("Final [%s]: mean=%.4f, max=%.4f", mode, cd.mean(), cd.max())

    # Save model
    torch.save(kwyk_model.state_dict(), EXP_DIR / "kwyk_fixed_warmstart.pth")
    log.info("Model saved to %s", EXP_DIR / "kwyk_fixed_warmstart.pth")


if __name__ == "__main__":
    main()
