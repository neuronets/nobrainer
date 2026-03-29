#!/usr/bin/env python
"""Compare new model predictions against original kwyk container.

Usage:
    python 05_compare_kwyk.py \
        --new-model checkpoints/bayesian \
        --kwyk-dir /path/to/kwyk \
        --manifest manifest.csv \
        --split test \
        --output-dir results/comparison
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess

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
        description="Compare new model vs original kwyk predictions",
    )
    parser.add_argument(
        "--new-model",
        type=str,
        required=True,
        help="Path to new model directory (model.pth + croissant.json)",
    )
    parser.add_argument(
        "--kwyk-dir",
        type=str,
        required=True,
        help="Path to original kwyk repository (containing kwyk/cli.py)",
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
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory for comparison outputs",
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
# Run original kwyk prediction
# ---------------------------------------------------------------------------
def run_kwyk_prediction(
    kwyk_dir: str,
    infile: str,
    outdir: Path,
) -> Path | None:
    """Run original kwyk CLI to produce a prediction.

    Calls::

        python {kwyk_dir}/kwyk/cli.py predict \\
            -m bwn_multi -n 1 {infile} {outprefix}

    Returns the path to the prediction NIfTI, or None on failure.
    """
    vol_stem = Path(infile).stem.replace(".nii", "")
    outprefix = str(outdir / f"kwyk_{vol_stem}")
    cmd = [
        "python",
        str(Path(kwyk_dir) / "kwyk" / "cli.py"),
        "predict",
        "-m",
        "bwn_multi",
        "-n",
        "1",
        infile,
        outprefix,
    ]
    log.info("Running kwyk: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0:
            log.error("kwyk failed (rc=%d): %s", result.returncode, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("kwyk timed out for %s", infile)
        return None
    except FileNotFoundError:
        log.error("kwyk CLI not found at %s", cmd[1])
        return None

    # kwyk outputs {outprefix}_means.nii.gz or {outprefix}.nii.gz
    for suffix in ["_means.nii.gz", ".nii.gz", "_prediction.nii.gz"]:
        candidate = Path(outprefix + suffix)
        if candidate.exists():
            return candidate

    log.warning("Could not find kwyk output for prefix %s", outprefix)
    return None


# ---------------------------------------------------------------------------
# Spatial correlation between uncertainty maps
# ---------------------------------------------------------------------------
def compute_spatial_correlation(map1: np.ndarray, map2: np.ndarray) -> float:
    """Compute Pearson correlation between two spatial maps.

    Parameters
    ----------
    map1, map2 : np.ndarray
        Flattened or volumetric arrays of the same shape.

    Returns
    -------
    float
        Pearson correlation coefficient, or 0.0 on failure.
    """
    v1 = map1.flatten().astype(np.float64)
    v2 = map2.flatten().astype(np.float64)

    # Remove positions where both are zero
    mask = (v1 != 0) | (v2 != 0)
    if mask.sum() < 2:
        return 0.0

    v1 = v1[mask]
    v2 = v2[mask]

    std1 = np.std(v1)
    std2 = np.std(v2)
    if std1 == 0 or std2 == 0:
        return 0.0

    return float(np.corrcoef(v1, v2)[0, 1])


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------
def plot_dice_scatter(
    kwyk_dices: list[float],
    new_dices: list[float],
    volume_names: list[str],
    output_path: Path,
) -> None:
    """Generate scatter plot: kwyk Dice (x) vs new model Dice (y)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(kwyk_dices, new_dices, alpha=0.7, edgecolors="k", s=50)

    # Identity line
    lims = [0.0, 1.0]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y = x")

    ax.set_xlabel("Original kwyk Dice")
    ax.set_ylabel("New Model Dice")
    ax.set_title("Dice Comparison: Original kwyk vs New Model")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    log.info("Scatter plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Compare new model vs original kwyk on test volumes."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kwyk_pred_dir = output_dir / "kwyk_predictions"
    kwyk_pred_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load new model -----------------------------------------------------
    from nobrainer.processing.segmentation import Segmentation

    log.info("Loading new model from %s", args.new_model)
    seg = Segmentation.load(args.new_model)
    block_shape = seg.block_shape_ or (32, 32, 32)

    # ---- Load manifest ------------------------------------------------------
    pairs = load_manifest(args.manifest, split=args.split)
    log.info(
        "Comparing on %d volumes from split '%s'",
        len(pairs),
        args.split,
    )

    if not pairs:
        log.error("No volumes found for split '%s'. Exiting.", args.split)
        return

    # ---- Evaluate each volume -----------------------------------------------
    results: list[dict] = []
    kwyk_dices: list[float] = []
    new_dices: list[float] = []
    volume_names: list[str] = []

    for idx, (img_path, lbl_path) in enumerate(pairs):
        vol_name = Path(img_path).stem
        log.info("Volume %d/%d: %s", idx + 1, len(pairs), vol_name)

        # Load ground truth and binarize
        gt_arr = np.asarray(nib.load(lbl_path).dataobj, dtype=np.float32)
        gt_binary = (gt_arr > 0).astype(np.float32)

        # ---- New model prediction -------------------------------------------
        label_img = seg.predict(img_path, block_shape=block_shape)
        new_pred = np.asarray(label_img.dataobj, dtype=np.float32)
        new_binary = (new_pred > 0).astype(np.float32)
        new_dice = compute_dice(new_binary, gt_binary)
        log.info("  New model Dice = %.4f", new_dice)

        # ---- Original kwyk prediction ---------------------------------------
        kwyk_pred_path = run_kwyk_prediction(args.kwyk_dir, img_path, kwyk_pred_dir)
        kwyk_dice = float("nan")
        if kwyk_pred_path is not None and kwyk_pred_path.exists():
            kwyk_arr = np.asarray(
                nib.load(str(kwyk_pred_path)).dataobj, dtype=np.float32
            )
            kwyk_binary = (kwyk_arr > 0).astype(np.float32)
            kwyk_dice = compute_dice(kwyk_binary, gt_binary)
            log.info("  kwyk Dice      = %.4f", kwyk_dice)
        else:
            log.warning("  kwyk prediction not available for %s", vol_name)

        results.append(
            {
                "volume": vol_name,
                "new_dice": new_dice,
                "kwyk_dice": kwyk_dice,
                "image_path": img_path,
            }
        )
        new_dices.append(new_dice)
        kwyk_dices.append(kwyk_dice)
        volume_names.append(vol_name)

    # ---- Save comparison CSV ------------------------------------------------
    csv_path = output_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["volume", "new_dice", "kwyk_dice", "image_path"],
        )
        writer.writeheader()
        writer.writerows(results)
    log.info("Comparison table saved to %s", csv_path)

    # ---- Scatter plot -------------------------------------------------------
    # Filter out NaN kwyk dices for plotting
    valid_mask = [not np.isnan(kd) for kd in kwyk_dices]
    valid_kwyk = [kd for kd, v in zip(kwyk_dices, valid_mask) if v]
    valid_new = [nd for nd, v in zip(new_dices, valid_mask) if v]
    valid_names = [n for n, v in zip(volume_names, valid_mask) if v]

    if valid_kwyk:
        scatter_path = output_dir / "dice_scatter.png"
        plot_dice_scatter(valid_kwyk, valid_new, valid_names, scatter_path)
    else:
        log.warning("No valid kwyk predictions; skipping scatter plot")

    # ---- Spatial correlation of uncertainty maps ----------------------------
    # Check if both models have uncertainty outputs
    new_results_dir = Path(args.new_model).parent / "results"
    if new_results_dir.exists():
        log.info("Checking for uncertainty map correlations...")
        for vol_name in volume_names:
            new_var_path = new_results_dir / f"{vol_name}_variance.nii.gz"
            kwyk_var_candidates = list(kwyk_pred_dir.glob(f"kwyk_{vol_name}*variance*"))
            if new_var_path.exists() and kwyk_var_candidates:
                new_var = np.asarray(
                    nib.load(str(new_var_path)).dataobj, dtype=np.float32
                )
                kwyk_var = np.asarray(
                    nib.load(str(kwyk_var_candidates[0])).dataobj,
                    dtype=np.float32,
                )
                corr = compute_spatial_correlation(new_var, kwyk_var)
                log.info(
                    "  %s uncertainty correlation: %.4f",
                    vol_name,
                    corr,
                )

    # ---- Summary ------------------------------------------------------------
    log.info("=" * 60)
    log.info("Comparison Summary (%s split)", args.split)
    log.info("  Volumes compared  : %d", len(results))
    if valid_kwyk:
        log.info("  Mean kwyk Dice    : %.4f", np.nanmean(kwyk_dices))
    log.info("  Mean new Dice     : %.4f", np.mean(new_dices))
    if valid_kwyk:
        improvement = np.mean(valid_new) - np.mean(valid_kwyk)
        log.info("  Mean improvement  : %+.4f", improvement)
    log.info("  Output directory  : %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
