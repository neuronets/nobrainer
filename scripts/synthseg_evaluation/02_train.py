#!/usr/bin/env python
"""Train a segmentation model with real, synthetic, or mixed data.

Usage:
    python 02_train.py --config config.yaml --mode real --model unet
    python 02_train.py --config config.yaml --mode mixed --model swin_unetr
    python 02_train.py --config config.yaml --mode synthetic --model kwyk_meshnet
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train with SynthSeg evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["real", "synthetic", "mixed"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--zarr-store",
        default=None,
        help="OME-Zarr store path (alternative to --manifest for zarr data)",
    )
    parser.add_argument(
        "--zarr-level", type=int, default=0, help="Pyramid level (default 0)"
    )
    return parser.parse_args()


def load_manifest(path, split):
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                pairs.append((row["t1w_path"], row["label_path"]))
    return pairs


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    synth_cfg = config["synthseg"]
    train_cfg = config["training"]

    epochs = args.epochs or train_cfg["epochs"]
    n_classes = data_cfg["n_classes"]
    block_shape = tuple(data_cfg["block_shape"])
    batch_size = data_cfg["batch_size"]
    lr = train_cfg["lr"]
    label_mapping = data_cfg["label_mapping"]

    output_dir = Path(args.output_dir) / f"{args.model}_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    from nobrainer.processing.dataset import Dataset
    from nobrainer.processing.segmentation import Segmentation

    # Build dataset — from zarr store or NIfTI manifest
    if args.zarr_store:
        log.info(
            "Loading from zarr store: %s (level=%d)", args.zarr_store, args.zarr_level
        )
        train_pairs = []  # not needed for zarr path
        ds = (
            Dataset.from_zarr(
                args.zarr_store,
                block_shape=block_shape,
                n_classes=n_classes,
                partition="train",
                level=args.zarr_level,
            )
            .batch(batch_size)
            .binarize(label_mapping)
            .augment(train_cfg.get("augmentation_profile", "standard"))
        )
        n_volumes = len(ds.data)
    else:
        train_pairs = load_manifest(args.manifest, "train")
        n_volumes = len(train_pairs)
        ds = (
            Dataset.from_files(
                train_pairs, block_shape=block_shape, n_classes=n_classes
            )
            .batch(batch_size)
            .binarize(label_mapping)
            .augment(train_cfg.get("augmentation_profile", "standard"))
        )

    log.info(
        "Training: mode=%s, model=%s, %d volumes, %d epochs",
        args.mode,
        args.model,
        n_volumes,
        epochs,
    )

    if args.mode in ("synthetic", "mixed"):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        # SynthSeg can read labels from zarr store or NIfTI paths
        if args.zarr_store:
            gen = SynthSegGenerator(
                zarr_store=args.zarr_store,
                zarr_level=args.zarr_level,
                n_samples_per_map=synth_cfg["n_samples_per_map"],
                elastic_std=synth_cfg["elastic_std"],
                rotation_range=synth_cfg["rotation_range"],
                scaling_bounds=synth_cfg["scaling_bounds"],
                flipping=synth_cfg["flipping"],
                randomize_resolution=synth_cfg["randomize_resolution"],
                resolution_range=tuple(synth_cfg["resolution_range"]),
                bias_field_std=synth_cfg["bias_field_std"],
                noise_std=synth_cfg["noise_std"],
                intensity_prior=tuple(synth_cfg["intensity_prior"]),
                std_prior=tuple(synth_cfg["std_prior"]),
            )
        else:
            label_paths = [p[1] for p in train_pairs]
            gen = SynthSegGenerator(
                label_paths,
                n_samples_per_map=synth_cfg["n_samples_per_map"],
                elastic_std=synth_cfg["elastic_std"],
                rotation_range=synth_cfg["rotation_range"],
                scaling_bounds=synth_cfg["scaling_bounds"],
                flipping=synth_cfg["flipping"],
                randomize_resolution=synth_cfg["randomize_resolution"],
                resolution_range=tuple(synth_cfg["resolution_range"]),
                bias_field_std=synth_cfg["bias_field_std"],
                noise_std=synth_cfg["noise_std"],
                intensity_prior=tuple(synth_cfg["intensity_prior"]),
                std_prior=tuple(synth_cfg["std_prior"]),
            )
        if args.mode == "mixed":
            ds = ds.mix(gen, ratio=train_cfg["mixed_ratio"])
            log.info("Mixed mode: %.0f%% synthetic", train_cfg["mixed_ratio"] * 100)

    # Build model
    model_args = {"n_classes": n_classes}
    if args.model in ("swin_unetr", "segresnet"):
        model_args["feature_size"] = 12 if args.model == "swin_unetr" else 16

    seg = Segmentation(
        args.model, model_args=model_args, checkpoint_filepath=str(output_dir)
    )

    # Experiment tracking
    from nobrainer.experiment import ExperimentTracker

    tracker = ExperimentTracker(
        output_dir=output_dir,
        config={
            "mode": args.mode,
            "model": args.model,
            "epochs": epochs,
            "n_classes": n_classes,
            "batch_size": batch_size,
        },
        project="synthseg-evaluation",
        name=f"{args.model}_{args.mode}",
    )

    import torch

    seg.fit(
        ds,
        epochs=epochs,
        optimizer=torch.optim.Adam,
        opt_args={"lr": lr},
        callbacks=[tracker.callback(mode=args.mode, model=args.model)],
    )
    seg.save(output_dir)
    tracker.finish()
    log.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
