#!/usr/bin/env python
"""Evaluate a trained segmentation model on test volumes.

Computes per-class Dice for each volume (matching McClure et al. 2019,
Section 2.4.1, Eq. 19), then averages across classes per volume.  The
reported "class Dice" in Table 3 of the paper is the mean ± std of
these per-volume average Dice scores.

For Bayesian models, MC inference produces variance and entropy maps
(Eq. 20) saved as NIfTI files.

Usage:
    python 04_evaluate.py --model checkpoints/bvwn_multi_prior \
        --manifest manifest.csv --split test --n-samples 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from utils import load_config, save_figure, setup_logging

log = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def load_manifest(manifest_path: str, split: str) -> list[tuple[str, str]]:
    pairs = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                pairs.append((row["t1w_path"], row["label_path"]))
    return pairs


def per_class_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute Dice coefficient for each class c = 1..n_classes-1.

    Matches Eq. 19 in McClure et al. (2019):
        Dice_c = 2*TP_c / (2*TP_c + FN_c + FP_c)

    Class 0 (background / unknown) is excluded, matching the paper:
    "averaging across all output voxels not classified as background".

    Parameters
    ----------
    pred : np.ndarray
        Integer label predictions.
    gt : np.ndarray
        Integer ground truth labels.
    n_classes : int
        Total number of classes (including background).

    Returns
    -------
    np.ndarray
        Shape ``(n_classes - 1,)`` — Dice for classes 1..n_classes-1.
    """
    dice_scores = np.zeros(n_classes - 1)
    for c in range(1, n_classes):
        pred_c = (pred == c).astype(np.float64)
        gt_c = (gt == c).astype(np.float64)
        intersection = (pred_c * gt_c).sum()
        total = pred_c.sum() + gt_c.sum()
        if total > 0:
            dice_scores[c - 1] = 2.0 * intersection / total
        else:
            # Both empty for this class — perfect agreement
            dice_scores[c - 1] = 1.0
    return dice_scores


def compute_entropy(prob_map: np.ndarray) -> np.ndarray:
    """Compute entropy of softmax probabilities (Eq. 20).

    H(y|x) = -sum_c p(y_c|x) log p(y_c|x)
    """
    eps = 1e-10
    return -(prob_map * np.log(prob_map + eps)).sum(axis=0)


def plot_prediction_overlay(
    t1w_arr: np.ndarray,
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    output_path: Path,
    title: str = "Prediction Overlay",
) -> None:
    """3-panel figure: T1w, prediction, ground truth (middle axial slice)."""
    mid = t1w_arr.shape[2] // 2
    t1 = t1w_arr[:, :, mid]
    pred = pred_arr[:, :, mid]
    gt = gt_arr[:, :, mid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1.T, cmap="gray", origin="lower")
    axes[0].set_title("T1w Input")
    axes[0].axis("off")

    axes[1].imshow(t1.T, cmap="gray", origin="lower")
    axes[1].imshow(
        pred.T,
        cmap="nipy_spectral",
        alpha=0.4,
        origin="lower",
        vmin=0,
        vmax=max(pred.max(), 1),
    )
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(t1.T, cmap="gray", origin="lower")
    axes[2].imshow(
        gt.T,
        cmap="nipy_spectral",
        alpha=0.4,
        origin="lower",
        vmin=0,
        vmax=max(gt.max(), 1),
    )
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_per_class_dice(
    class_dice_all: np.ndarray,
    class_names: list[str] | None,
    output_path: Path,
) -> None:
    """Bar chart of mean per-class Dice across all volumes."""
    mean_dice = class_dice_all.mean(axis=0)
    std_dice = class_dice_all.std(axis=0)
    n = len(mean_dice)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.3), 6))
    x = np.arange(n)
    ax.bar(x, mean_dice, yerr=std_dice, capsize=2, alpha=0.7, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Dice")
    ax.set_title("Per-Class Dice (mean ± std across volumes)")
    ax.set_ylim(0, 1.05)
    if class_names and len(class_names) == n:
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config for n_classes and label_mapping ------------------------
    config = load_config(args.config)
    n_classes = config.get("n_classes", 2)
    label_mapping = config.get("label_mapping", "binary")

    # Load label names for plots
    class_names = None
    if label_mapping and label_mapping != "binary":
        mapping_path = (
            Path(__file__).parent / "label_mappings" / f"{label_mapping}-mapping.csv"
        )
        if mapping_path.exists():
            with open(mapping_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # Build class_names indexed by 'new' column (skip background=0)
            name_map = {}
            for r in rows:
                new_id = int(r["new"])
                if new_id > 0 and new_id not in name_map:
                    name_map[new_id] = r.get("label", str(new_id))
            class_names = [name_map.get(i, str(i)) for i in range(1, n_classes)]

    # Load remap function for ground truth
    remap_fn = None
    if label_mapping and label_mapping != "binary":
        from nobrainer.processing.dataset import _load_label_mapping

        remap_fn = _load_label_mapping(label_mapping)

    # ---- Load model ---------------------------------------------------------
    from nobrainer.processing.segmentation import Segmentation

    log.info("Loading model from %s", args.model)
    seg = Segmentation.load(args.model)
    block_shape = seg.block_shape_ or tuple(config["block_shape"])
    log.info(
        "Model: %s, block_shape=%s, n_classes=%s",
        seg.base_model,
        block_shape,
        n_classes,
    )

    # ---- Load manifest ------------------------------------------------------
    pairs = load_manifest(args.manifest, split=args.split)
    log.info("Evaluating %d volumes from split '%s'", len(pairs), args.split)

    if not pairs:
        log.error("No volumes for split '%s'", args.split)
        return

    # ---- Evaluate each volume -----------------------------------------------
    results: list[dict] = []
    all_class_dice: list[np.ndarray] = []
    n_samples = args.n_samples

    for idx, (img_path, lbl_path) in enumerate(pairs):
        vol_name = Path(img_path).stem
        log.info("Volume %d/%d: %s", idx + 1, len(pairs), vol_name)

        # Load and remap ground truth
        gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
        if remap_fn is not None:
            gt_arr = remap_fn(gt_arr)
        elif label_mapping == "binary":
            gt_arr = (gt_arr > 0).astype(np.int32)

        t1w_arr = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)

        # Predict
        if n_samples > 0:
            pred_result = seg.predict(
                img_path, block_shape=block_shape, n_samples=n_samples
            )
            if isinstance(pred_result, tuple):
                label_img, var_img, entropy_img = pred_result
                nib.save(var_img, str(output_dir / f"{vol_name}_variance.nii.gz"))
                nib.save(entropy_img, str(output_dir / f"{vol_name}_entropy.nii.gz"))
            else:
                label_img = pred_result
        else:
            label_img = seg.predict(img_path, block_shape=block_shape)

        pred_arr = np.asarray(label_img.dataobj, dtype=np.int32)

        # Per-class Dice (Eq. 19)
        class_dice = per_class_dice(pred_arr, gt_arr, n_classes)
        avg_dice = float(class_dice.mean())
        all_class_dice.append(class_dice)

        log.info(
            "  Avg class Dice = %.4f (min=%.4f, max=%.4f)",
            avg_dice,
            class_dice.min(),
            class_dice.max(),
        )

        results.append(
            {
                "volume": vol_name,
                "image_path": img_path,
                "avg_class_dice": avg_dice,
                "min_class_dice": float(class_dice.min()),
                "max_class_dice": float(class_dice.max()),
            }
        )

        # Overlay figure
        plot_prediction_overlay(
            t1w_arr,
            pred_arr.astype(np.float32),
            gt_arr.astype(np.float32),
            fig_dir / f"{vol_name}_overlay.png",
            title=f"{vol_name} — Avg Dice={avg_dice:.4f}",
        )

    # ---- Per-class Dice bar chart -------------------------------------------
    class_dice_matrix = np.array(all_class_dice)  # (n_volumes, n_classes-1)
    plot_per_class_dice(class_dice_matrix, class_names, fig_dir / "per_class_dice.png")

    # ---- Save CSV with per-volume results -----------------------------------
    csv_path = output_dir / "dice_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "volume",
                "image_path",
                "avg_class_dice",
                "min_class_dice",
                "max_class_dice",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # ---- Save per-class Dice matrix -----------------------------------------
    np.save(output_dir / "per_class_dice.npy", class_dice_matrix)

    # ---- Summary (matching Table 3 format) ----------------------------------
    avg_dices = [r["avg_class_dice"] for r in results]
    log.info("=" * 60)
    log.info("Evaluation Summary (%s split, %d-class)", args.split, n_classes)
    log.info("  Volumes           : %d", len(avg_dices))
    log.info("  MC samples        : %d", n_samples)
    log.info("  Class Dice        : %.4f ± %.4f", np.mean(avg_dices), np.std(avg_dices))
    log.info("  Min volume Dice   : %.4f", np.min(avg_dices))
    log.info("  Max volume Dice   : %.4f", np.max(avg_dices))
    if class_names:
        mean_per_class = class_dice_matrix.mean(axis=0)
        worst_5 = np.argsort(mean_per_class)[:5]
        log.info(
            "  Worst 5 classes   : %s",
            ", ".join(f"{class_names[i]}={mean_per_class[i]:.3f}" for i in worst_5),
        )
    log.info("  Output            : %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
