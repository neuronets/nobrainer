#!/usr/bin/env python
"""Sweep over block sizes to compare segmentation performance.

Usage:
    python 06_block_size_sweep.py --manifest manifest.csv --config config.yaml
    python 06_block_size_sweep.py --manifest manifest.csv --config config.yaml \
        --block-sizes 32 64 128 --epochs 20 --output-dir results/sweep
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Block size sweep for Bayesian MeshNet segmentation",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the dataset manifest CSV",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Block sizes to sweep over (default: 32 64 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs per block size (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sweep",
        help="Directory for sweep outputs",
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
# Train + evaluate for one block size
# ---------------------------------------------------------------------------
def train_and_evaluate(
    block_size: int,
    config: dict,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    epochs: int,
) -> dict:
    """Train a Bayesian MeshNet at the given block size and evaluate Dice.

    Returns
    -------
    dict
        Keys: block_size, mean_dice, std_dice, per_volume_dices, final_loss.
    """
    import nibabel as nib

    from nobrainer.models import get as get_model
    from nobrainer.models.bayesian.utils import accumulate_kl
    from nobrainer.prediction import predict
    from nobrainer.processing.dataset import Dataset

    block_shape = (block_size, block_size, block_size)
    n_classes = config["n_classes"]
    batch_size = config["batch_size"]
    lr = config.get("lr", 1e-4)
    kl_weight = config.get("kl_weight", 1.0)

    log.info(
        "Training with block_size=%d for %d epochs...",
        block_size,
        epochs,
    )

    # Build dataset
    ds_train = (
        Dataset.from_files(train_pairs, block_shape=block_shape, n_classes=n_classes)
        .batch(batch_size)
        .binarize(config.get("label_mapping", "binary"))
    )
    train_loader = ds_train.dataloader

    # Build model
    model_args = {
        "n_classes": n_classes,
        "filters": config.get("filters", 96),
        "receptive_field": config.get("receptive_field", 37),
        "dropout_rate": config.get("dropout_rate", 0.25),
    }
    model = get_model("bayesian_meshnet")(**model_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer.zero_grad()
            pred = model(images)
            loss = ce_loss(pred, labels) + kl_weight * accumulate_kl(model)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            log.info(
                "  block_size=%d, epoch %d/%d, loss=%.6f",
                block_size,
                epoch + 1,
                epochs,
                final_loss,
            )

    # Evaluate on validation set
    model.eval()
    per_volume_dices: list[float] = []

    for img_path, lbl_path in val_pairs:
        gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.float32)
        gt_binary = (gt_arr > 0).astype(np.float32)

        pred_img = predict(
            inputs=img_path,
            model=model,
            block_shape=block_shape,
            batch_size=4,
            return_labels=True,
        )
        pred_arr = np.asarray(pred_img.dataobj, dtype=np.float32)
        pred_binary = (pred_arr > 0).astype(np.float32)

        dice = compute_dice(pred_binary, gt_binary)
        per_volume_dices.append(dice)

    mean_dice = float(np.mean(per_volume_dices)) if per_volume_dices else 0.0
    std_dice = float(np.std(per_volume_dices)) if per_volume_dices else 0.0

    log.info(
        "  block_size=%d: Dice=%.4f +/- %.4f (%d volumes)",
        block_size,
        mean_dice,
        std_dice,
        len(per_volume_dices),
    )

    return {
        "block_size": block_size,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "per_volume_dices": per_volume_dices,
        "final_loss": final_loss,
    }


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------
def plot_block_size_comparison(
    sweep_results: list[dict],
    output_path: Path,
) -> None:
    """Generate bar chart: block_size on x, Dice on y with error bars."""
    block_sizes = [r["block_size"] for r in sweep_results]
    means = [r["mean_dice"] for r in sweep_results]
    stds = [r["std_dice"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(block_sizes))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xlabel("Block Size")
    ax.set_ylabel("Dice Score")
    ax.set_title("Block Size Sweep — Bayesian MeshNet")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in block_sizes])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with mean values
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    log.info("Bar chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run block size sweep and generate comparison outputs."""
    args = parse_args()
    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load config --------------------------------------------------------
    config = load_config(args.config)
    log.info("Config loaded from %s", args.config)
    log.info(
        "Block size sweep: sizes=%s, epochs=%d",
        args.block_sizes,
        args.epochs,
    )

    # ---- Load manifest ------------------------------------------------------
    train_pairs = load_manifest(args.manifest, split="train")
    val_pairs = load_manifest(args.manifest, split="val")
    log.info(
        "Manifest: %d train, %d val volumes",
        len(train_pairs),
        len(val_pairs),
    )

    if not train_pairs:
        log.error("No training volumes found. Exiting.")
        return
    if not val_pairs:
        log.warning("No validation volumes found; Dice will be empty.")

    # ---- Run sweep ----------------------------------------------------------
    sweep_results: list[dict] = []

    for block_size in args.block_sizes:
        result = train_and_evaluate(
            block_size=block_size,
            config=config,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            epochs=args.epochs,
        )
        sweep_results.append(result)

    # ---- Save comparison CSV ------------------------------------------------
    csv_path = output_dir / "block_size_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["block_size", "mean_dice", "std_dice", "final_loss"],
        )
        writer.writeheader()
        for r in sweep_results:
            writer.writerow(
                {
                    "block_size": r["block_size"],
                    "mean_dice": r["mean_dice"],
                    "std_dice": r["std_dice"],
                    "final_loss": r["final_loss"],
                }
            )
    log.info("Comparison CSV saved to %s", csv_path)

    # ---- Bar chart ----------------------------------------------------------
    chart_path = output_dir / "block_size_comparison.png"
    plot_block_size_comparison(sweep_results, chart_path)

    # ---- Summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    best = max(sweep_results, key=lambda r: r["mean_dice"])
    log.info("=" * 60)
    log.info("Block Size Sweep Complete")
    log.info("  Block sizes tested: %s", args.block_sizes)
    log.info("  Epochs per size   : %d", args.epochs)
    log.info(
        "  Best block size   : %d (Dice=%.4f)", best["block_size"], best["mean_dice"]
    )
    for r in sweep_results:
        log.info(
            "  block_size=%3d: Dice=%.4f +/- %.4f, loss=%.6f",
            r["block_size"],
            r["mean_dice"],
            r["std_dice"],
            r["final_loss"],
        )
    log.info("  Output directory  : %s", output_dir)
    log.info("  Elapsed time      : %.1f s", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
