#!/usr/bin/env python
"""Train a Bayesian MeshNet with optional warm-start from deterministic weights.

Usage:
    # With warm-start from a trained deterministic MeshNet
    python 03_train_bayesian.py --manifest manifest.csv --config config.yaml \
        --warmstart checkpoints/meshnet

    # Without warm-start (train from scratch)
    python 03_train_bayesian.py --manifest manifest.csv --config config.yaml \
        --no-warmstart

    # Override epochs
    python 03_train_bayesian.py --manifest manifest.csv --config config.yaml \
        --warmstart checkpoints/meshnet --epochs 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils import compute_dice, load_config, save_figure, setup_logging

log = setup_logging(__name__)


# ---------------------------------------------------------------------------
# ELBO loss: CrossEntropy + KL divergence from Bayesian layers
# ---------------------------------------------------------------------------
class ELBOLoss(nn.Module):
    """Evidence Lower Bound loss combining CE and KL divergence.

    Parameters
    ----------
    model : nn.Module
        Bayesian model whose layers carry ``.kl`` attributes after
        each forward pass.
    kl_weight : float
        Scaling factor for the KL term.  ``1.0`` corresponds to the
        standard variational free-energy; smaller values down-weight
        the regularisation (cold posterior).
    """

    def __init__(self, model: nn.Module, kl_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = model
        self.kl_weight = kl_weight
        self._last_kl: float = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from nobrainer.models.bayesian.utils import accumulate_kl

        ce_loss = self.ce(pred, target)
        kl_loss = accumulate_kl(self.model)
        self._last_kl = kl_loss.item()
        return ce_loss + self.kl_weight * kl_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Bayesian MeshNet with optional warm-start",
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
        "--output-dir",
        type=str,
        default="checkpoints/bayesian",
        help="Directory for saving model checkpoints and figures",
    )
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Path to a trained deterministic MeshNet directory (containing model.pth)",
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Explicitly disable warm-start (train from scratch)",
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


# ---------------------------------------------------------------------------
# Validation with MC inference
# ---------------------------------------------------------------------------
def evaluate_mc_dice(
    model: nn.Module,
    val_pairs: list[tuple[str, str]],
    block_shape: tuple[int, int, int],
    n_samples: int,
    label_mapping: str | None,
) -> tuple[list[float], list[float]]:
    """Run MC inference on each validation volume.

    Returns
    -------
    mean_dices : list[float]
        Mean Dice across MC samples for each volume.
    std_dices : list[float]
        Std of Dice across MC samples for each volume.
    """
    from nobrainer.prediction import predict

    mean_dices: list[float] = []
    std_dices: list[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for img_path, lbl_path in val_pairs:
        import nibabel as nib

        gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.float32)
        if label_mapping is None or label_mapping == "binary":
            gt_arr = (gt_arr > 0).astype(np.float32)

        # Multiple stochastic forward passes
        sample_dices: list[float] = []
        for s in range(n_samples):
            # Keep model in train mode for stochastic sampling
            model.train()
            pred_img = predict(
                inputs=img_path,
                model=model,
                block_shape=block_shape,
                batch_size=4,
                return_labels=True,
            )
            pred_arr = np.asarray(pred_img.dataobj).astype(np.float32)
            if label_mapping is None or label_mapping == "binary":
                pred_arr = (pred_arr > 0).astype(np.float32)

            dice = compute_dice(pred_arr, gt_arr)
            sample_dices.append(dice)

        vol_mean = float(np.mean(sample_dices))
        vol_std = float(np.std(sample_dices))
        mean_dices.append(vol_mean)
        std_dices.append(vol_std)
        log.info(
            "  Val volume %s: MC Dice=%.4f +/- %.4f (%d samples)",
            Path(img_path).name,
            vol_mean,
            vol_std,
            n_samples,
        )

    return mean_dices, std_dices


# ---------------------------------------------------------------------------
# Learning curve with uncertainty bands
# ---------------------------------------------------------------------------
def plot_learning_curve(
    train_losses: list[float],
    val_losses: list[float],
    val_dice_means: list[float],
    val_dice_stds: list[float],
    kl_terms: list[float],
    output_path: Path,
) -> None:
    """Generate learning curve with uncertainty bands.

    Left y-axis: train loss, val loss, KL term.
    Right y-axis: mean MC Dice with +/- std shading.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax_loss = plt.subplots(figsize=(12, 7))
    ax_dice = ax_loss.twinx()

    # Loss curves
    ax_loss.plot(epochs, train_losses, "b-", label="Train Loss (ELBO)")
    if val_losses:
        ax_loss.plot(epochs, val_losses, "b--", alpha=0.7, label="Val Loss")
    if kl_terms:
        ax_loss.plot(epochs, kl_terms, "g-.", alpha=0.6, label="KL Term")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss / KL", color="b")
    ax_loss.tick_params(axis="y", labelcolor="b")

    # Dice with uncertainty bands
    if val_dice_means:
        means = np.array(val_dice_means)
        stds = np.array(val_dice_stds)
        dice_epochs = list(range(1, len(means) + 1))
        ax_dice.plot(
            dice_epochs, means, "r-o", markersize=3, label="Val MC Dice (mean)"
        )
        ax_dice.fill_between(
            dice_epochs,
            np.clip(means - stds, 0, 1),
            np.clip(means + stds, 0, 1),
            color="r",
            alpha=0.15,
            label="Val MC Dice (+/- std)",
        )
        ax_dice.set_ylabel("Dice Score", color="r")
        ax_dice.tick_params(axis="y", labelcolor="r")
        ax_dice.set_ylim(0.0, 1.0)

    fig.suptitle("Bayesian MeshNet Training — ELBO Loss & MC Dice")
    fig.tight_layout()

    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_dice, labels_dice = ax_dice.get_legend_handles_labels()
    ax_loss.legend(
        lines_loss + lines_dice,
        labels_loss + labels_dice,
        loc="center right",
    )

    save_figure(fig, output_path)
    plt.close(fig)
    log.info("Learning curve saved to %s", output_path)


# ---------------------------------------------------------------------------
# Training loop (lower-level, using nobrainer.training.fit)
# ---------------------------------------------------------------------------
def train_bayesian(
    model: nn.Module,
    train_loader,
    val_loader,
    elbo_loss: ELBOLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    val_pairs: list[tuple[str, str]],
    block_shape: tuple[int, int, int],
    n_samples: int,
    label_mapping: str | None,
    checkpoint_dir: Path,
) -> dict:
    """Custom training loop for Bayesian MeshNet with ELBO loss.

    We use a custom per-epoch loop to track KL divergence and perform
    periodic MC validation with uncertainty estimation.
    """
    # Accumulators for plotting
    train_losses: list[float] = []
    val_losses_list: list[float] = []
    val_dice_means: list[float] = []
    val_dice_stds: list[float] = []
    kl_terms: list[float] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_loss = float("inf")

    for epoch in range(epochs):
        t_epoch = time.time()

        # -- Train one epoch --------------------------------------------------
        model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
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

            # Squeeze channel dim from labels if present
            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer.zero_grad()
            pred = model(images)
            loss = elbo_loss(pred, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl += elbo_loss._last_kl
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        train_losses.append(avg_loss)
        kl_terms.append(avg_kl)

        # -- Validate ---------------------------------------------------------
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
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

                    pred = model(images)
                    loss = elbo_loss(pred, labels)
                    val_loss += loss.item()
                    n_val += 1
            val_loss = val_loss / max(n_val, 1)
        val_losses_list.append(val_loss)

        # -- MC Dice evaluation (every 10 epochs or last epoch) ---------------
        if val_pairs and (epoch == epochs - 1 or (epoch + 1) % 10 == 0):
            mean_dices, std_dices = evaluate_mc_dice(
                model, val_pairs, block_shape, n_samples, label_mapping
            )
            overall_mean = float(np.mean(mean_dices)) if mean_dices else 0.0
            overall_std = float(np.mean(std_dices)) if std_dices else 0.0
            val_dice_means.append(overall_mean)
            val_dice_stds.append(overall_std)
        else:
            # Carry forward last value for continuity in plots
            if val_dice_means:
                val_dice_means.append(val_dice_means[-1])
                val_dice_stds.append(val_dice_stds[-1])
            else:
                val_dice_means.append(float("nan"))
                val_dice_stds.append(float("nan"))

        # -- Checkpoint best --------------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)

        elapsed = time.time() - t_epoch
        log.info(
            "Epoch %d/%d: train_loss=%.6f val_loss=%.6f kl=%.6f " "dice=%.4f (%.1fs)",
            epoch + 1,
            epochs,
            avg_loss,
            val_loss,
            avg_kl,
            val_dice_means[-1] if val_dice_means else 0.0,
            elapsed,
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses_list,
        "val_dice_means": val_dice_means,
        "val_dice_stds": val_dice_stds,
        "kl_terms": kl_terms,
        "best_loss": best_loss,
        "epochs_completed": epochs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Train Bayesian MeshNet with optional warm-start."""
    args = parse_args()
    t_start = time.time()

    # ---- Load config --------------------------------------------------------
    config = load_config(args.config)
    epochs = (
        args.epochs if args.epochs is not None else config.get("bayesian_epochs", 50)
    )
    n_classes = config["n_classes"]
    block_shape = tuple(config["block_shape"])
    batch_size = config["batch_size"]
    lr = config.get("lr", 1e-4)
    kl_weight = config.get("kl_weight", 1.0)
    initial_rho = config.get("initial_rho", -3.0)
    n_samples = config.get("n_samples", 10)
    label_mapping = config.get("label_mapping", "binary")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_warmstart = args.warmstart is not None and not args.no_warmstart

    log.info("Config loaded from %s", args.config)
    log.info(
        "Training Bayesian MeshNet: epochs=%d, n_classes=%d, "
        "block_shape=%s, kl_weight=%.4f, warmstart=%s",
        epochs,
        n_classes,
        block_shape,
        kl_weight,
        use_warmstart,
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
    train_loader = ds_train.dataloader

    ds_val = None
    val_loader = None
    if val_pairs:
        ds_val = (
            Dataset.from_files(val_pairs, block_shape=block_shape, n_classes=n_classes)
            .batch(batch_size)
            .binarize(label_mapping)
        )
        val_loader = ds_val.dataloader

    # ---- Build Bayesian MeshNet ---------------------------------------------
    from nobrainer.models import get as get_model

    model_args = {
        "n_classes": n_classes,
        "filters": config.get("filters", 96),
        "receptive_field": config.get("receptive_field", 37),
        "dropout_rate": config.get("dropout_rate", 0.25),
    }
    log.info("Model args: %s", model_args)

    bayesian_model = get_model("bayesian_meshnet")(**model_args)
    log.info(
        "Bayesian MeshNet created: %d parameters",
        sum(p.numel() for p in bayesian_model.parameters()),
    )

    # ---- Optional warm-start ------------------------------------------------
    if use_warmstart:
        warmstart_dir = Path(args.warmstart)
        det_weights_path = warmstart_dir / "model.pth"

        if not det_weights_path.exists():
            log.error(
                "Warm-start weights not found at %s. "
                "Train a deterministic MeshNet first with 02_train_meshnet.py.",
                det_weights_path,
            )
            return

        log.info("Loading deterministic weights from %s", det_weights_path)

        # Build a deterministic MeshNet with matching architecture
        det_model = get_model("meshnet")(**model_args)
        det_model.load_state_dict(torch.load(det_weights_path, weights_only=True))

        from nobrainer.models.bayesian.warmstart import (
            warmstart_bayesian_from_deterministic,
        )

        n_transferred = warmstart_bayesian_from_deterministic(
            bayesian_model,
            det_model,
            initial_rho=initial_rho,
        )
        log.info("Warm-started %d layers from deterministic model", n_transferred)
        del det_model  # free memory
    else:
        log.info("Training Bayesian MeshNet from scratch (no warm-start)")

    # ---- ELBO loss and optimiser --------------------------------------------
    elbo_loss = ELBOLoss(bayesian_model, kl_weight=kl_weight)
    optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=lr)

    # ---- Train --------------------------------------------------------------
    result = train_bayesian(
        model=bayesian_model,
        train_loader=train_loader,
        val_loader=val_loader,
        elbo_loss=elbo_loss,
        optimizer=optimizer,
        epochs=epochs,
        val_pairs=val_pairs,
        block_shape=block_shape,
        n_samples=n_samples,
        label_mapping=label_mapping,
        checkpoint_dir=output_dir,
    )

    # ---- Learning curve with uncertainty bands ------------------------------
    fig_path = output_dir / "learning_curve.png"
    plot_learning_curve(
        train_losses=result["train_losses"],
        val_losses=result["val_losses"],
        val_dice_means=result["val_dice_means"],
        val_dice_stds=result["val_dice_stds"],
        kl_terms=result["kl_terms"],
        output_path=fig_path,
    )

    # ---- Save with Croissant-ML metadata ------------------------------------
    # Save final weights
    torch.save(bayesian_model.state_dict(), output_dir / "model.pth")

    # Use Segmentation estimator's save for Croissant metadata
    from nobrainer.processing.segmentation import Segmentation

    seg = Segmentation(
        base_model="bayesian_meshnet",
        model_args=model_args,
    )
    seg.model_ = bayesian_model
    seg.block_shape_ = block_shape
    seg.n_classes_ = n_classes
    seg._optimizer_class = "Adam"
    seg._optimizer_args = {"lr": lr}
    seg._loss_name = "ELBOLoss"
    seg._training_result = {
        "final_loss": result["train_losses"][-1] if result["train_losses"] else 0.0,
        "best_loss": result["best_loss"],
        "epochs_completed": result["epochs_completed"],
        "checkpoint_path": str(output_dir / "best_model.pth"),
    }
    seg._dataset = ds_train
    seg.save(output_dir)
    log.info("Model and Croissant-ML metadata saved to %s", output_dir)

    # ---- Summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    final_dice = (
        result["val_dice_means"][-1] if result["val_dice_means"] else float("nan")
    )
    log.info("=" * 60)
    log.info("Bayesian MeshNet training complete")
    log.info("  Output directory : %s", output_dir)
    log.info("  Epochs           : %d", epochs)
    log.info("  Warm-start       : %s", "yes" if use_warmstart else "no")
    log.info("  KL weight        : %.4f", kl_weight)
    log.info(
        "  Final train loss : %.6f",
        result["train_losses"][-1] if result["train_losses"] else 0.0,
    )
    log.info("  Best train loss  : %.6f", result["best_loss"])
    log.info("  Val MC Dice      : %.4f", final_dice)
    log.info("  MC samples       : %d", n_samples)
    log.info("  Elapsed time     : %.1f s", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
