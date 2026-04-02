"""Progressive GAN reproduction script using OME-Zarr pyramidal stores.

Trains a ProgressiveGAN that reads from progressively finer pyramid
levels as the network grows, matching the progressive training paradigm.

Usage::

    python scripts/progressive_gan/train.py --data brain.ome.zarr --output-dir ckpts/
    python scripts/progressive_gan/train.py --resume ckpts/latest.pt
    python scripts/progressive_gan/train.py --generate 10 --resume ckpts/final.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pyramid-to-GAN level mapping (T025)
# ---------------------------------------------------------------------------

MIN_GAN_RESOLUTION = 4  # GAN's minimum block size


def map_pyramid_to_gan_stages(
    n_pyramid_levels: int,
    resolution_schedule: list[int],
    volume_shape: tuple[int, int, int],
) -> list[int]:
    """Map GAN growth stages to pyramid levels.

    Stages proceed coarsest → finest.  If the pyramid has fewer levels
    than the GAN has stages, the finest available level is reused for
    remaining stages.  Levels whose resolution is smaller than the GAN's
    minimum (4³) are skipped.

    Parameters
    ----------
    n_pyramid_levels : int
        Number of pyramid levels in the store.
    resolution_schedule : list[int]
        GAN resolution schedule (e.g. [4, 8, 16, 32, 64]).
    volume_shape : tuple of int
        Full-resolution spatial shape (D, H, W).

    Returns
    -------
    list[int]
        Pyramid level for each GAN stage (length = len(resolution_schedule)).
    """
    n_stages = len(resolution_schedule)

    # Available levels: 0 (full) to n_pyramid_levels-1 (coarsest)
    # Filter out levels where shape < MIN_GAN_RESOLUTION
    valid_levels = []
    for lvl in range(n_pyramid_levels):
        factor = 2**lvl
        lvl_min_dim = min(s // factor for s in volume_shape)
        if lvl_min_dim >= MIN_GAN_RESOLUTION:
            valid_levels.append(lvl)

    if not valid_levels:
        log.warning(
            "No pyramid levels have resolution >= %d; using level 0",
            MIN_GAN_RESOLUTION,
        )
        return [0] * n_stages

    # Map stages: stage 0 (coarsest GAN res) → highest valid level number
    # stage N-1 (finest GAN res) → level 0
    mapping = []
    for stage_idx in range(n_stages):
        # Linear mapping: stage 0 → coarsest valid, last stage → finest (0)
        # Interpolate stage index into valid_levels range
        if n_stages == 1:
            lvl_idx = 0  # single stage → finest level
        elif len(valid_levels) >= n_stages:
            # More levels than stages: evenly sample from coarsest to finest
            frac = stage_idx / (n_stages - 1)  # 0.0 → 1.0
            lvl_idx = round((1.0 - frac) * (len(valid_levels) - 1))
        else:
            # Fewer levels than stages: reuse finest for extra stages
            lvl_idx = max(0, len(valid_levels) - 1 - stage_idx)
        mapping.append(valid_levels[lvl_idx])

    skipped = n_pyramid_levels - len(valid_levels)
    if skipped > 0:
        log.warning(
            "Skipped %d pyramid level(s) with resolution < %d",
            skipped,
            MIN_GAN_RESOLUTION,
        )
    if len(valid_levels) < n_stages:
        log.warning(
            "Pyramid has %d valid levels but GAN has %d stages; "
            "reusing finest level for remaining stages",
            len(valid_levels),
            n_stages,
        )

    return mapping


# ---------------------------------------------------------------------------
# Training loop (T026)
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from nobrainer.datasets.zarr_store import store_info
    from nobrainer.models.generative.progressivegan import ProgressiveGAN
    from nobrainer.processing.dataset import PatchDataset

    config = _load_config(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store info
    info = store_info(config["data_path"])
    n_levels = info.get("n_levels", 1)
    volume_shape = tuple(info["volume_shape"])
    log.info(
        "Store: %s (%d subjects, %d levels, shape=%s)",
        config["data_path"],
        info["n_subjects"],
        n_levels,
        volume_shape,
    )

    # Build resolution schedule from volume shape
    min_dim = min(volume_shape)
    resolution_schedule = []
    res = MIN_GAN_RESOLUTION
    while res <= min_dim:
        resolution_schedule.append(res)
        res *= 2
    log.info("Resolution schedule: %s", resolution_schedule)

    # Map pyramid levels to GAN stages
    level_mapping = map_pyramid_to_gan_stages(
        n_levels, resolution_schedule, volume_shape
    )
    log.info("Level mapping (stage → pyramid level): %s", level_mapping)

    # Create model
    model = ProgressiveGAN(
        latent_size=config["latent_dim"],
        fmap_base=config["fmap_base"],
        fmap_max=config["fmap_max"],
        resolution_schedule=resolution_schedule,
        steps_per_phase=config["steps_per_phase"],
        lambda_gp=config["lambda_gp"],
        lr=config["learning_rate"],
    )

    # Resume from checkpoint
    start_stage = 0
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model._step_count = ckpt.get("step_count", 0)
        start_stage = ckpt.get("stage", 0)
        log.info(
            "Resumed from %s (stage=%d, step=%d)",
            args.resume,
            start_stage,
            model._step_count,
        )

    # Train each stage
    for stage_idx in range(start_stage, len(resolution_schedule)):
        pyramid_level = level_mapping[stage_idx]
        target_res = resolution_schedule[stage_idx]
        log.info(
            "=== Stage %d/%d: resolution=%d³, pyramid level=%d ===",
            stage_idx + 1,
            len(resolution_schedule),
            target_res,
            pyramid_level,
        )

        # Build dataset from pyramid level
        from nobrainer.processing import Dataset

        ds = Dataset.from_zarr(
            config["data_path"],
            block_shape=(target_res, target_res, target_res),
            level=pyramid_level,
        )
        patch_ds = PatchDataset(
            data=ds.data,
            block_shape=(target_res, target_res, target_res),
            patches_per_volume=4,
        )
        loader = DataLoader(
            patch_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # Train
        trainer = pl.Trainer(
            max_steps=config["steps_per_phase"],
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=True,
            logger=False,
        )
        trainer.fit(model, loader)

        # Save checkpoint
        ckpt_path = output_dir / f"stage_{stage_idx:02d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "step_count": model._step_count,
                "stage": stage_idx + 1,
                "resolution_schedule": resolution_schedule,
                "level_mapping": level_mapping,
                "config": config,
            },
            ckpt_path,
        )
        # Also save as latest
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "step_count": model._step_count,
                "stage": stage_idx + 1,
                "resolution_schedule": resolution_schedule,
                "level_mapping": level_mapping,
                "config": config,
            },
            output_dir / "latest.pt",
        )
        log.info("Checkpoint saved: %s", ckpt_path)

    # Save final
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step_count": model._step_count,
            "stage": len(resolution_schedule),
            "resolution_schedule": resolution_schedule,
            "level_mapping": level_mapping,
            "config": config,
        },
        output_dir / "final.pt",
    )
    log.info("Training complete. Final checkpoint: %s", output_dir / "final.pt")


# ---------------------------------------------------------------------------
# Generation (T027)
# ---------------------------------------------------------------------------


def generate(args: argparse.Namespace) -> None:
    """Generate synthetic volumes from a trained checkpoint."""
    import nibabel as nib
    import numpy as np

    from nobrainer.models.generative.progressivegan import ProgressiveGAN

    if not args.resume:
        log.error("--resume required for generation")
        sys.exit(1)

    ckpt = torch.load(args.resume, weights_only=False)
    config = ckpt["config"]
    resolution_schedule = ckpt["resolution_schedule"]

    model = ProgressiveGAN(
        latent_size=config["latent_dim"],
        fmap_base=config["fmap_base"],
        fmap_max=config["fmap_max"],
        resolution_schedule=resolution_schedule,
        steps_per_phase=config["steps_per_phase"],
        lambda_gp=config["lambda_gp"],
        lr=config["learning_rate"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Set to final level
    model.generator.current_level = len(resolution_schedule) - 1
    model.generator.alpha = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n = args.generate
    output_dir = Path(config["output_dir"]) / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating %d volumes...", n)
    with torch.no_grad():
        z = torch.randn(n, config["latent_dim"], device=device)
        volumes = model.generator(z)  # (N, 1, D, H, W)

    for i in range(n):
        vol = volumes[i, 0].cpu().numpy().astype(np.float32)
        img = nib.Nifti1Image(vol, np.eye(4))
        out_path = output_dir / f"generated_{i:04d}.nii.gz"
        nib.save(img, str(out_path))
        log.info("Saved: %s (shape=%s)", out_path, vol.shape)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> dict:
    """Load config from YAML, override with CLI args."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    for key in [
        "data_path",
        "output_dir",
        "latent_dim",
        "steps_per_phase",
        "batch_size",
        "learning_rate",
        "lambda_gp",
        "checkpoint_freq",
        "fmap_base",
        "fmap_max",
    ]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            config[key] = cli_val

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Progressive GAN training with OME-Zarr pyramidal stores"
    )
    parser.add_argument("--data", dest="data_path", help="OME-Zarr store path")
    parser.add_argument("--output-dir", help="Checkpoint output directory")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    parser.add_argument(
        "--generate", type=int, default=0, help="Generate N volumes (requires --resume)"
    )
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--steps-per-phase", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lambda-gp", type=float)
    parser.add_argument("--checkpoint-freq", type=int)

    args = parser.parse_args()

    if args.generate > 0:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
