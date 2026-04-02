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

    if not args.resume:
        log.error("--resume required for generation")
        sys.exit(1)

    model, config, _ckpt = _load_model_from_checkpoint(args.resume)
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
# Evaluation
# ---------------------------------------------------------------------------


def _load_model_from_checkpoint(ckpt_path: str):
    """Load a ProgressiveGAN from a checkpoint file."""
    from nobrainer.models.generative.progressivegan import ProgressiveGAN

    ckpt = torch.load(ckpt_path, weights_only=False)
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
    model.generator.current_level = len(resolution_schedule) - 1
    model.generator.alpha = 1.0
    return model, config, ckpt


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation checks on a trained GAN.

    1. **Memorization check**: generate N volumes and compare each to
       every training volume.  Reports the maximum cosine similarity
       and the Euclidean distance to the nearest training sample.  A
       well-trained GAN should produce novel volumes (low similarity).

    2. **Interpolation smoothness**: linearly interpolate between pairs
       of random latent vectors and measure the L2 distance between
       successive generated volumes.  A smooth latent space produces
       roughly constant step sizes; memorization or mode collapse
       produces step-like jumps (high variance in step sizes).
    """
    import numpy as np

    if not args.resume:
        log.error("--resume required for evaluation")
        sys.exit(1)

    model, config, ckpt = _load_model_from_checkpoint(args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(config["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_eval = getattr(args, "n_eval", 16) or 16
    n_interp_steps = getattr(args, "n_interp_steps", 10) or 10
    n_interp_pairs = getattr(args, "n_interp_pairs", 5) or 5
    latent_dim = config["latent_dim"]

    # ------------------------------------------------------------------
    # 1. Memorization check
    # ------------------------------------------------------------------
    log.info("=== Memorization check (%d generated vs training data) ===", n_eval)

    # Load training volumes from the zarr store
    if config.get("data_path"):
        from nobrainer.datasets.zarr_store import store_info

        info = store_info(config["data_path"])
        import zarr

        store = zarr.open_group(config["data_path"], mode="r")
        n_train = info["n_subjects"]
        # Load all training volumes at level 0 (flattened for comparison)
        train_vols = []
        img_arr = store["images/0"]
        is_bf16 = img_arr.attrs.get("_nobrainer_dtype") == "bfloat16"
        for i in range(min(n_train, 100)):  # cap at 100 for memory
            vol = np.asarray(img_arr[i]).astype(np.float32)
            if is_bf16:
                from nobrainer.datasets.zarr_store import decode_bfloat16

                vol = decode_bfloat16(np.asarray(img_arr[i]).astype(np.uint16))
            train_vols.append(vol.ravel())
        train_matrix = np.stack(train_vols)  # (N_train, D*H*W)
        train_norms = np.linalg.norm(train_matrix, axis=1, keepdims=True)
        train_normed = train_matrix / np.maximum(train_norms, 1e-8)
    else:
        log.warning("No data_path in config; skipping memorization check")
        train_normed = None

    # Generate evaluation volumes
    with torch.no_grad():
        z = torch.randn(n_eval, latent_dim, device=device)
        gen_vols = model.generator(z).cpu().numpy()  # (N, 1, D, H, W)

    if train_normed is not None:
        max_cosine_sims = []
        min_l2_dists = []
        for i in range(n_eval):
            gv = gen_vols[i, 0].ravel().astype(np.float32)
            gv_norm = np.linalg.norm(gv)
            gv_normed = gv / max(gv_norm, 1e-8)

            # Cosine similarity to each training volume
            cosines = train_normed @ gv_normed
            max_cos = float(cosines.max())
            max_cosine_sims.append(max_cos)

            # L2 distance to nearest training volume
            diffs = train_matrix - gv[None, :]
            l2s = np.linalg.norm(diffs, axis=1)
            min_l2_dists.append(float(l2s.min()))

        avg_max_cos = np.mean(max_cosine_sims)
        avg_min_l2 = np.mean(min_l2_dists)
        log.info(
            "Memorization: avg_max_cosine_sim=%.4f, avg_min_L2=%.4f",
            avg_max_cos,
            avg_min_l2,
        )
        log.info(
            "  (cosine > 0.95 suggests memorization; "
            "well-trained GAN typically < 0.8)"
        )

        memorization_results = {
            "n_generated": n_eval,
            "n_training": len(train_vols),
            "avg_max_cosine_similarity": float(avg_max_cos),
            "max_cosine_similarity": float(max(max_cosine_sims)),
            "avg_min_l2_distance": float(avg_min_l2),
            "min_l2_distance": float(min(min_l2_dists)),
        }
    else:
        memorization_results = {"skipped": True}

    # ------------------------------------------------------------------
    # 2. Interpolation smoothness
    # ------------------------------------------------------------------
    log.info(
        "=== Interpolation smoothness (%d pairs, %d steps) ===",
        n_interp_pairs,
        n_interp_steps,
    )

    pair_cvs = []  # coefficient of variation per pair
    pair_step_sizes = []

    with torch.no_grad():
        for pair_idx in range(n_interp_pairs):
            z_a = torch.randn(1, latent_dim, device=device)
            z_b = torch.randn(1, latent_dim, device=device)

            # Generate volumes along the interpolation path
            alphas = torch.linspace(0, 1, n_interp_steps, device=device)
            interp_vols = []
            for alpha in alphas:
                z_interp = (1 - alpha) * z_a + alpha * z_b
                vol = model.generator(z_interp)
                interp_vols.append(vol.cpu().numpy().ravel())

            # Compute step sizes (L2 between consecutive volumes)
            step_sizes = []
            for j in range(1, len(interp_vols)):
                d = np.linalg.norm(interp_vols[j] - interp_vols[j - 1])
                step_sizes.append(float(d))

            step_sizes = np.array(step_sizes)
            mean_step = step_sizes.mean()
            std_step = step_sizes.std()
            cv = std_step / max(mean_step, 1e-8)
            pair_cvs.append(cv)
            pair_step_sizes.append(step_sizes.tolist())

            log.info(
                "  Pair %d: mean_step=%.4f, std=%.4f, CV=%.4f %s",
                pair_idx,
                mean_step,
                std_step,
                cv,
                "(smooth)" if cv < 0.3 else "(step-like)" if cv > 0.7 else "",
            )

    avg_cv = float(np.mean(pair_cvs))
    log.info(
        "Interpolation: avg_CV=%.4f %s",
        avg_cv,
        (
            "(smooth latent space)"
            if avg_cv < 0.3
            else "(step-like — possible memorization)" if avg_cv > 0.7 else "(moderate)"
        ),
    )

    interpolation_results = {
        "n_pairs": n_interp_pairs,
        "n_steps": n_interp_steps,
        "avg_coefficient_of_variation": float(avg_cv),
        "per_pair_cv": [float(cv) for cv in pair_cvs],
        "interpretation": (
            "smooth"
            if avg_cv < 0.3
            else "step-like (possible memorization)" if avg_cv > 0.7 else "moderate"
        ),
    }

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    import json

    report = {
        "checkpoint": str(args.resume),
        "memorization": memorization_results,
        "interpolation_smoothness": interpolation_results,
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Evaluation report saved: %s", report_path)


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
        "--generate",
        type=int,
        default=0,
        help="Generate N volumes (requires --resume)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run memorization + interpolation evaluation (requires --resume)",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=16,
        help="Number of volumes for memorization check (default 16)",
    )
    parser.add_argument(
        "--n-interp-steps",
        type=int,
        default=10,
        help="Interpolation steps between latent pairs (default 10)",
    )
    parser.add_argument(
        "--n-interp-pairs",
        type=int,
        default=5,
        help="Number of latent pairs for interpolation (default 5)",
    )
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--steps-per-phase", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lambda-gp", type=float)
    parser.add_argument("--checkpoint-freq", type=int)

    args = parser.parse_args()

    if args.evaluate:
        evaluate(args)
    elif args.generate > 0:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
