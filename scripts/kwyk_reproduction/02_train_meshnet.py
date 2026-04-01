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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume from",
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
    n_classes: int = 2,
) -> list[float]:
    """Compute per-volume mean class Dice on validation set.

    Returns a list of mean Dice scores (averaged across classes), one per volume.
    """
    import nibabel as nib

    from nobrainer.prediction import predict
    from nobrainer.training import get_device

    # Load remap function for multi-class label mappings
    remap_fn = None
    if label_mapping and label_mapping != "binary":
        from nobrainer.processing.dataset import _load_label_mapping

        remap_fn = _load_label_mapping(label_mapping)

    dice_scores = []
    device = get_device()
    model = seg.model_.to(device)
    model.eval()

    for img_path, lbl_path in val_pairs:
        pred_img = predict(
            inputs=img_path,
            model=model,
            block_shape=block_shape,
            batch_size=128,
            return_labels=True,
        )
        pred_arr = np.asarray(pred_img.dataobj, dtype=np.int32)

        gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.int32)
        if remap_fn is not None:
            gt_arr = remap_fn(torch.from_numpy(gt_arr)).numpy()
        elif label_mapping is None or label_mapping == "binary":
            gt_arr = (gt_arr > 0).astype(np.int32)
            pred_arr = (pred_arr > 0).astype(np.int32)

        # Per-class Dice (skip background class 0)
        class_dices = []
        for c in range(1, n_classes):
            pred_c = pred_arr == c
            gt_c = gt_arr == c
            intersection = (pred_c & gt_c).sum()
            total = pred_c.sum() + gt_c.sum()
            class_dices.append(2.0 * intersection / total if total > 0 else 1.0)

        mean_dice = float(np.mean(class_dices))
        dice_scores.append(mean_dice)
        log.info(
            "  Val volume %s: Dice=%.4f",
            Path(img_path).name,
            mean_dice,
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

    # Auto-scale batch size to GPU memory
    from nobrainer.gpu import auto_batch_size as _auto_bs
    from nobrainer.gpu import gpu_count
    from nobrainer.processing.dataset import Dataset

    if gpu_count() > 0:
        from nobrainer.models import get as get_model

        _tmp_model = get_model("meshnet")(
            n_classes=n_classes,
            filters=config.get("filters", 96),
            receptive_field=config.get("receptive_field", 37),
            dropout_rate=config.get("dropout_rate", 0.25),
        )
        batch_size = _auto_bs(
            _tmp_model,
            block_shape,
            n_classes=n_classes,
            target_memory_fraction=0.90,
        )
        del _tmp_model
        log.info("Auto batch size: %d (target 90%% GPU memory)", batch_size)

    patches_per_volume = config.get("patches_per_volume", 50)
    zarr_store = config.get("zarr_store")

    if zarr_store and Path(zarr_store).exists():
        log.info("Using Zarr store: %s", zarr_store)
        ds_train = (
            Dataset.from_zarr(
                zarr_store,
                block_shape=block_shape,
                n_classes=n_classes,
                partition="train",
            )
            .batch(batch_size)
            .binarize(label_mapping)
            .streaming(patches_per_volume=patches_per_volume)
        )
    else:
        ds_train = (
            Dataset.from_files(
                train_pairs, block_shape=block_shape, n_classes=n_classes
            )
            .batch(batch_size)
            .binarize(label_mapping)
            .streaming(patches_per_volume=patches_per_volume)
        )

    n_train = len(ds_train.data) if hasattr(ds_train, "data") else len(train_pairs)
    log.info(
        "Training data: %d volumes × %d patches = %d blocks/epoch, batch_size=%d",
        n_train,
        patches_per_volume,
        n_train * patches_per_volume,
        batch_size,
    )

    # ---- Build validation dataset for per-epoch block-level metrics ----------
    ds_val = None
    if val_pairs:
        if zarr_store and Path(zarr_store).exists():
            ds_val = (
                Dataset.from_zarr(
                    zarr_store,
                    block_shape=block_shape,
                    n_classes=n_classes,
                    partition="val",
                )
                .batch(batch_size)
                .binarize(label_mapping)
                .streaming(patches_per_volume=patches_per_volume)
            )
        else:
            ds_val = (
                Dataset.from_files(
                    val_pairs, block_shape=block_shape, n_classes=n_classes
                )
                .batch(batch_size)
                .binarize(label_mapping)
                .streaming(patches_per_volume=patches_per_volume)
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

    val_dice_per_epoch: list[float] = []
    val_dice_freq = config.get("val_dice_freq", 5)

    # Simple logging callback (picklable — no closures)
    def _log_cb(epoch, logs, model):
        msg = f"Epoch {epoch + 1}/{epochs}: train_loss={logs['loss']:.6f}"
        if "val_loss" in logs:
            msg += f" val_loss={logs['val_loss']:.6f}"
        if "val_acc" in logs:
            msg += f" val_acc={logs['val_acc']:.4f}"
        if "val_bal_acc" in logs:
            msg += f" bal_acc={logs['val_bal_acc']:.4f}"
        log.info(msg)

    seg.fit(
        dataset_train=ds_train,
        dataset_validate=ds_val,
        epochs=epochs,
        optimizer=torch.optim.Adam,
        opt_args={"lr": lr},
        callbacks=[_log_cb],
        checkpoint_freq=val_dice_freq,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        model_parallel=config.get("model_parallel", False),
        resume_from=args.resume,
    )

    history = seg._training_result.get("history", [])

    if history:
        last = history[-1]
        log.info(
            "Training complete. %s",
            " ".join(f"{k}={v:.4f}" for k, v in last.items() if isinstance(v, float)),
        )
    else:
        log.info("Training complete (no history).")

    # Ensure model is on the right device after DDP
    from nobrainer.training import get_device

    seg.model_.to(get_device())

    # Evaluate full-volume Dice on each checkpointed epoch
    if val_pairs:
        for epoch_idx in range(len(history)):
            epoch_num = history[epoch_idx].get("epoch", epoch_idx + 1)
            ckpt_file = output_dir / f"epoch_{epoch_num:03d}.pth"
            if ckpt_file.exists():
                log.info("Evaluating Dice at epoch %d...", epoch_num)
                seg.model_.load_state_dict(
                    torch.load(ckpt_file, map_location=get_device(), weights_only=True)
                )
                dice_scores = evaluate_val_dice(
                    seg, val_pairs, block_shape, label_mapping, n_classes
                )
                mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
                history[epoch_idx]["val_dice"] = mean_dice
                log.info("  Epoch %d Dice: %.4f", epoch_num, mean_dice)

    train_losses = [h["loss"] for h in history]
    val_dice_per_epoch = [h.get("val_dice", float("nan")) for h in history]
    fig_path = output_dir / "learning_curve.png"
    plot_learning_curve(train_losses, val_dice_per_epoch, fig_path)

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
    if val_losses:
        log.info("  Final val loss   : %.6f", val_losses[-1])
    final_dice = [d for d in val_dice_per_epoch if not np.isnan(d)]
    if final_dice:
        log.info("  Val Dice (mean)  : %.4f", final_dice[-1])
    log.info("  Elapsed time     : %.1f s", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
