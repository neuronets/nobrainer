#!/usr/bin/env python
"""Compare results across training modes and models.

Usage:
    python 04_compare.py --results-dir checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for eval results: checkpoints/<model>_<mode>/eval/dice_scores.csv
    rows = []
    for eval_dir in sorted(results_dir.glob("*/eval")):
        csv_path = eval_dir / "dice_scores.csv"
        if not csv_path.exists():
            continue

        name = eval_dir.parent.name  # e.g., "unet_real"
        parts = name.rsplit("_", 1)
        model = parts[0] if len(parts) == 2 else name
        mode = parts[1] if len(parts) == 2 else "unknown"

        with open(csv_path) as f:
            scores = [float(r["avg_dice"]) for r in csv.DictReader(f)]

        if scores:
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "mean_dice": f"{np.mean(scores):.4f}",
                    "std_dice": f"{np.std(scores):.4f}",
                    "n_volumes": len(scores),
                }
            )
            log.info(
                "%s (%s): %.4f ± %.4f", model, mode, np.mean(scores), np.std(scores)
            )

    if not rows:
        log.warning("No results found in %s", results_dir)
        return

    # Write comparison table
    csv_path = output_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["model", "mode", "mean_dice", "std_dice", "n_volumes"])
        w.writeheader()
        w.writerows(rows)
    log.info("Comparison table: %s", csv_path)

    # Generate figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = sorted(set(r["model"] for r in rows))
        modes = sorted(set(r["mode"] for r in rows))
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 6))
        for i, mode in enumerate(modes):
            means = []
            stds = []
            for model in models:
                match = [r for r in rows if r["model"] == model and r["mode"] == mode]
                if match:
                    means.append(float(match[0]["mean_dice"]))
                    stds.append(float(match[0]["std_dice"]))
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + i * width, means, width, yerr=stds, label=mode, capsize=3)

        ax.set_xlabel("Model")
        ax.set_ylabel("Mean Class Dice")
        ax.set_title("SynthSeg Evaluation: Model × Training Mode")
        ax.set_xticks(x + width * (len(modes) - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "comparison_figure.png", dpi=150)
        plt.close(fig)
        log.info("Comparison figure: %s", output_dir / "comparison_figure.png")
    except ImportError:
        log.warning("matplotlib not available, skipping figure")


if __name__ == "__main__":
    main()
