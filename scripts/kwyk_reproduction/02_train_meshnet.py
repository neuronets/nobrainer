#!/usr/bin/env python
"""Train a deterministic MeshNet for brain extraction / parcellation.

Usage:
    python 02_train_meshnet.py --manifest manifest.csv --config config.yaml
    python 02_train_meshnet.py --manifest manifest.csv --config config.yaml \
        --output-dir checkpoints/meshnet --epochs 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import compute_dice, load_config, save_figure, setup_logging

log = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train deterministic MeshNet for brain segmentation",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the dataset manifest CSV (output of 01_assemble_dataset.py)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/meshnet",
        help="Directory for saving model checkpoints and figures",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs from config",
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


def evaluate_val_dice(
    seg,
    val_pairs: list[tuple[str, str]],
    block_shape: tuple[int, int, int],
    label_mapping: str | None,
) -> list[float]:
    """Compute per-volume Dice on validation set.

    Returns a list of Dice scores, one per validation volume.
    """
    from nobrainer.prediction import predict

    dice_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = seg.model_.to(device)
    model.eval()

    for img_path, lbl_path in val_pairs:
        # Predict labels for this volume
        pred_img = predict(
            inputs=img_path,
            model=model,
            block_shape=block_shape,
            batch_size=4,
            return_labels=True,
        )
        pred_arr = np.asarray(pred_img.dataobj).astype(np.float32)

        # Load ground-truth and apply same binarisation
        import nibabel as nib

        gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.float32)
        if label_mapping is None or label_mapping == "binary":
            gt_arr = (gt_arr > 0).astype(np.float32)
            pred_arr = (pred_arr > 0).astype(np.float32)

        dice = compute_dice(pred_arr, gt_arr)
        dice_scores.append(dice)
        log.info(
            "  Val volume %s: Dice=%.4f",
            Path(img_path).name,
            dice,
        )

    return dice_scores


def plot_learning_curve(
    train_losses: list[float],
    val_dice_scores: list[float],
    output_path: Path,
) -> None:
    """Generate dual y-axis learning curve (loss left, Dice right)."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax_loss = plt.subplots(figsize=(10, 6))
    ax_dice = ax_loss.twinx()

    ax_loss.plot(epochs, train_losses, "b-", label="Train Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="b")
    ax_loss.tick_params(axis="y", labelcolor="b")

    if val_dice_scores:
        ax_dice.plot(epochs, val_dice_scores, "r-o", label="Val Dice (mean)")
        ax_dice.set_ylabel("Dice Score", color="r")
        ax_dice.tick_params(axis="y", labelcolor="r")
        ax_dice.set_ylim(0.0, 1.0)

    fig.suptitle("MeshNet Training — Loss & Validation Dice")
    fig.tight_layout()

    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_dice, labels_dice = ax_dice.get_legend_handles_labels()
    ax_loss.legend(
        lines_loss + lines_dice, labels_loss + labels_dice, loc="center right"
    )

    save_figure(fig, output_path)
    plt.close(fig)
    log.info("Learning curve saved to %s", output_path)


def main() -> None:
    """Train deterministic MeshNet and evaluate on validation set."""
    args = parse_args()
    t_start = time.time()

    # ---- Load config --------------------------------------------------------
    config = load_config(args.config)
    epochs = (
        args.epochs if args.epochs is not None else config.get("pretrain_epochs", 50)
    )
    n_classes = config["n_classes"]
    block_shape = tuple(config["block_shape"])
    batch_size = config["batch_size"]
    lr = config.get("lr", 1e-4)
    label_mapping = config.get("label_mapping", "binary")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Config loaded from %s", args.config)
    log.info(
        "Training MeshNet: epochs=%d, n_classes=%d, block_shape=%s, batch_size=%d",
        epochs,
        n_classes,
        block_shape,
        batch_size,
    )

    # ---- Load manifest and build datasets -----------------------------------
    train_pairs = load_manifest(args.manifest, split="train")
    val_pairs = load_manifest(args.manifest, split="val")
    log.info("Manifest: %d train, %d val volumes", len(train_pairs), len(val_pairs))

    if not train_pairs:
        log.error("No training volumes found in manifest. Exiting.")
        return

    from nobrainer.processing.dataset import Dataset

    ds_train = (
        Dataset.from_files(train_pairs, block_shape=block_shape, n_classes=n_classes)
        .batch(batch_size)
        .binarize(label_mapping)
    )

    # ---- Train with Segmentation estimator ----------------------------------
    from nobrainer.processing.segmentation import Segmentation

    model_args = {
        "n_classes": n_classes,
        "filters": config.get("filters", 96),
        "receptive_field": config.get("receptive_field", 37),
        "dropout_rate": config.get("dropout_rate", 0.25),
    }

    log.info("Model args: %s", model_args)

    seg = Segmentation(
        base_model="meshnet",
        model_args=model_args,
        checkpoint_filepath=str(output_dir),
    )

    # Collect per-epoch losses via a callback
    train_losses: list[float] = []

    def _record_loss(epoch: int, loss: float, model: torch.nn.Module) -> None:
        train_losses.append(loss)
        log.info("Epoch %d/%d: train_loss=%.6f", epoch + 1, epochs, loss)

    seg.fit(
        dataset_train=ds_train,
        epochs=epochs,
        optimizer=torch.optim.Adam,
        opt_args={"lr": lr},
        callbacks=[_record_loss],
    )

    log.info(
        "Training complete. Final loss=%.6f", train_losses[-1] if train_losses else 0.0
    )

    # ---- Evaluate on validation set -----------------------------------------
    val_dice_per_epoch: list[float] = []
    if val_pairs:
        log.info("Evaluating on %d validation volumes...", len(val_pairs))
        dice_scores = evaluate_val_dice(seg, val_pairs, block_shape, label_mapping)
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        val_dice_per_epoch = [mean_dice]  # single evaluation at end
        log.info(
            "Validation Dice: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            mean_dice,
            float(np.std(dice_scores)) if dice_scores else 0.0,
            float(np.min(dice_scores)) if dice_scores else 0.0,
            float(np.max(dice_scores)) if dice_scores else 0.0,
        )

    # ---- Learning curve figure ----------------------------------------------
    # For the Dice axis, pad with NaN so lengths match (evaluated only at end)
    val_dice_for_plot = [float("nan")] * (len(train_losses) - 1) + val_dice_per_epoch
    if len(val_dice_for_plot) < len(train_losses):
        val_dice_for_plot = val_dice_for_plot + [float("nan")] * (
            len(train_losses) - len(val_dice_for_plot)
        )

    fig_path = output_dir / "learning_curve.png"
    plot_learning_curve(train_losses, val_dice_for_plot, fig_path)

    # ---- Save model with Croissant-ML metadata ------------------------------
    seg.save(output_dir)
    log.info("Model and Croissant-ML metadata saved to %s", output_dir)

    # ---- Summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("MeshNet training complete")
    log.info("  Output directory : %s", output_dir)
    log.info("  Epochs           : %d", epochs)
    log.info("  Final train loss : %.6f", train_losses[-1] if train_losses else 0.0)
    if val_dice_per_epoch:
        log.info("  Val Dice (mean)  : %.4f", val_dice_per_epoch[-1])
    log.info("  Elapsed time     : %.1f s", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
