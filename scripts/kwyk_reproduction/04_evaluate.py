#!/usr/bin/env python
"""Evaluate a trained segmentation model on test volumes.

Usage:
    python 04_evaluate.py --model checkpoints/bayesian --manifest manifest.csv
    python 04_evaluate.py --model checkpoints/bayesian --manifest manifest.csv \
        --split test --n-samples 10 --output-dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from utils import compute_dice, save_figure, setup_logging

log = setup_logging(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained segmentation model on test volumes",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved model directory (containing model.pth + croissant.json)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the dataset manifest CSV",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of MC inference samples (0 = deterministic)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for evaluation outputs",
    )
    return parser.parse_args()


def load_manifest(manifest_path: str, split: str) -> list[tuple[str, str]]:
    """Load manifest CSV and return (image, label) pairs for the given split."""
    pairs = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                pairs.append((row["t1w_path"], row["label_path"]))
    return pairs


# ---------------------------------------------------------------------------
# Prediction overlay figure
# ---------------------------------------------------------------------------
def plot_prediction_overlay(
    t1w_arr: np.ndarray,
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    output_path: Path,
    title: str = "Prediction Overlay",
) -> None:
    """Generate a 3-panel figure: T1w, prediction overlay, ground truth overlay.

    Shows the middle axial slice for each panel.
    """
    mid = t1w_arr.shape[2] // 2
    t1_slice = t1w_arr[:, :, mid]
    pred_slice = pred_arr[:, :, mid]
    gt_slice = gt_arr[:, :, mid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: T1w
    axes[0].imshow(t1_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("T1w Input")
    axes[0].axis("off")

    # Panel 2: T1w + prediction overlay
    axes[1].imshow(t1_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(pred_slice.T, cmap="Reds", alpha=0.4, origin="lower")
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    # Panel 3: T1w + ground truth overlay
    axes[2].imshow(t1_slice.T, cmap="gray", origin="lower")
    axes[2].imshow(gt_slice.T, cmap="Greens", alpha=0.4, origin="lower")
    axes[2].set_title("Ground Truth Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Evaluate model on test volumes and save results."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ---------------------------------------------------------
    from nobrainer.processing.segmentation import Segmentation

    log.info("Loading model from %s", args.model)
    seg = Segmentation.load(args.model)
    block_shape = seg.block_shape_ or (32, 32, 32)
    log.info(
        "Model loaded: %s, block_shape=%s, n_classes=%s",
        seg.base_model,
        block_shape,
        seg.n_classes_,
    )

    # ---- Load manifest ------------------------------------------------------
    pairs = load_manifest(args.manifest, split=args.split)
    log.info(
        "Evaluating on %d volumes from split '%s'",
        len(pairs),
        args.split,
    )

    if not pairs:
        log.error("No volumes found for split '%s'. Exiting.", args.split)
        return

    # ---- Evaluate each volume -----------------------------------------------
    results: list[dict] = []
    n_samples = args.n_samples

    for idx, (img_path, lbl_path) in enumerate(pairs):
        vol_name = Path(img_path).stem
        log.info("Processing volume %d/%d: %s", idx + 1, len(pairs), vol_name)

        # Load ground truth and binarize
        gt_img = nib.load(lbl_path)
        gt_arr = np.asarray(gt_img.dataobj, dtype=np.float32)
        gt_binary = (gt_arr > 0).astype(np.float32)

        # Load T1w for overlay figure
        t1w_img = nib.load(img_path)
        t1w_arr = np.asarray(t1w_img.dataobj, dtype=np.float32)

        if n_samples > 0:
            # MC inference with uncertainty
            label_img, var_img, entropy_img = seg.predict(
                img_path,
                block_shape=block_shape,
                n_samples=n_samples,
            )
            pred_arr = np.asarray(label_img.dataobj, dtype=np.float32)
            pred_binary = (pred_arr > 0).astype(np.float32)

            # Save variance and entropy maps as NIfTI
            var_path = output_dir / f"{vol_name}_variance.nii.gz"
            nib.save(var_img, str(var_path))
            log.info("  Saved variance map: %s", var_path)

            entropy_path = output_dir / f"{vol_name}_entropy.nii.gz"
            nib.save(entropy_img, str(entropy_path))
            log.info("  Saved entropy map: %s", entropy_path)
        else:
            # Deterministic prediction
            label_img = seg.predict(img_path, block_shape=block_shape)
            pred_arr = np.asarray(label_img.dataobj, dtype=np.float32)
            pred_binary = (pred_arr > 0).astype(np.float32)

        # Compute Dice
        dice = compute_dice(pred_binary, gt_binary)
        log.info("  Dice = %.4f", dice)

        results.append({"volume": vol_name, "image_path": img_path, "dice": dice})

        # Generate prediction overlay figure
        fig_path = fig_dir / f"{vol_name}_overlay.png"
        plot_prediction_overlay(
            t1w_arr,
            pred_binary,
            gt_binary,
            fig_path,
            title=f"{vol_name} — Dice={dice:.4f}",
        )
        log.info("  Saved overlay figure: %s", fig_path)

    # ---- Save per-volume Dice CSV -------------------------------------------
    csv_path = output_dir / "dice_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["volume", "image_path", "dice"])
        writer.writeheader()
        writer.writerows(results)
    log.info("Per-volume Dice scores saved to %s", csv_path)

    # ---- Summary statistics -------------------------------------------------
    dice_values = [r["dice"] for r in results]
    log.info("=" * 60)
    log.info("Evaluation Summary (%s split)", args.split)
    log.info("  Volumes evaluated : %d", len(dice_values))
    log.info("  MC samples        : %d", n_samples)
    log.info("  Mean Dice         : %.4f", np.mean(dice_values))
    log.info("  Std Dice          : %.4f", np.std(dice_values))
    log.info("  Min Dice          : %.4f", np.min(dice_values))
    log.info("  Max Dice          : %.4f", np.max(dice_values))
    log.info("  Median Dice       : %.4f", np.median(dice_values))
    log.info("  Output directory  : %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
