"""Main command-line interface for nobrainer."""

from __future__ import annotations

import datetime
import os
import platform
import sys

import click
import nibabel as nib
import numpy as np
import torch

from .. import __version__
from ..prediction import predict as _predict
from ..training import get_device

_option_kwds = {"show_default": True}


class JSONParamType(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        try:
            import json

            return json.loads(value)
        except Exception:
            self.fail(f"{value} is not valid JSON", param, ctx)


@click.group()
@click.version_option(__version__, message="%(prog)s version %(version)s")
def cli():
    """A framework for developing neural network models for 3D image processing."""
    return


@cli.command()
@click.argument("infile")
@click.argument("outfile")
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to PyTorch model file (.pth) or model name.",
    **_option_kwds,
)
@click.option(
    "--model-type",
    default="unet",
    help=(
        "Model architecture: unet, vnet, attention_unet, unetr, meshnet, "
        "highresnet, bayesian_vnet, bayesian_meshnet."
    ),
    **_option_kwds,
)
@click.option(
    "--n-classes",
    type=int,
    default=1,
    help="Number of output classes.",
    **_option_kwds,
)
@click.option(
    "--in-channels",
    type=int,
    default=1,
    help="Number of input channels.",
    **_option_kwds,
)
@click.option(
    "-b",
    "--block-shape",
    default=(128, 128, 128),
    type=int,
    nargs=3,
    help="Shape of sub-volumes on which to predict.",
    **_option_kwds,
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Number of blocks to process per forward pass.",
    **_option_kwds,
)
@click.option(
    "--n-samples",
    type=int,
    default=1,
    help="Monte-Carlo samples for Bayesian uncertainty estimation (>1 enables MC-Dropout).",
    **_option_kwds,
)
@click.option(
    "--device",
    default="auto",
    help='Compute device: "auto", "cpu", "cuda", "cuda:0", …',
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress messages.", **_option_kwds
)
def predict(
    *,
    infile,
    outfile,
    model,
    model_type,
    n_classes,
    in_channels,
    block_shape,
    batch_size,
    n_samples,
    device,
    verbose,
):
    """Predict labels from a NIfTI volume using a trained PyTorch model.

    The predictions are saved to OUTFILE.
    """
    if os.path.exists(outfile):
        raise FileExistsError(f"Output file already exists: {outfile}")

    # Resolve device
    if device == "auto":
        _device = get_device()
    else:
        _device = torch.device(device)

    if verbose:
        click.echo(f"Using device: {_device}")

    # Load model architecture + weights
    from ..models import get as _get_model

    try:
        factory = _get_model(model_type)
        pt_model = factory(n_classes=n_classes, in_channels=in_channels)
        state = torch.load(model, map_location=_device, weights_only=True)
        pt_model.load_state_dict(state, strict=False)
    except Exception as exc:
        click.echo(click.style(f"ERROR: could not load model: {exc}", fg="red"))
        raise SystemExit(1) from exc

    if verbose:
        click.echo("Running prediction ...")

    if n_samples > 1:
        from ..prediction import predict_with_uncertainty

        try:
            label_img, var_img, entropy_img = predict_with_uncertainty(
                infile,
                pt_model,
                n_samples=n_samples,
                block_shape=block_shape,
                batch_size=batch_size,
                device=_device,
            )
            nib.save(label_img, outfile)
            nib.save(var_img, outfile.replace(".nii", "_var.nii"))
            nib.save(entropy_img, outfile.replace(".nii", "_entropy.nii"))
        except NotImplementedError:
            click.echo(
                click.style(
                    "predict_with_uncertainty not yet implemented; "
                    "falling back to deterministic predict()",
                    fg="yellow",
                )
            )
            out_img = _predict(
                infile,
                pt_model,
                block_shape=block_shape,
                batch_size=batch_size,
                device=_device,
            )
            nib.save(out_img, outfile)
    else:
        out_img = _predict(
            infile,
            pt_model,
            block_shape=block_shape,
            batch_size=batch_size,
            device=_device,
        )
        nib.save(out_img, outfile)

    if verbose:
        click.echo(click.style(f"Output saved to {outfile}", fg="green"))


@cli.command()
@click.option(
    "-i",
    "--input",
    "input_paths",
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help="TFRecord file(s) to convert.",
    **_option_kwds,
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for NIfTI or HDF5 files.",
    **_option_kwds,
)
@click.option(
    "--format",
    "output_format",
    default="nifti",
    type=click.Choice(["nifti", "hdf5"]),
    help="Output format.",
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress messages.", **_option_kwds
)
def convert_tfrecords(*, input_paths, output_dir, output_format, verbose):
    """Convert TFRecord files to NIfTI or HDF5 (no TensorFlow required)."""
    from ..io import convert_tfrecords as _convert

    if verbose:
        click.echo(f"Converting {len(input_paths)} TFRecord file(s) …")

    out_paths = _convert(
        tfrecord_paths=list(input_paths),
        output_dir=output_dir,
        output_format=output_format,
    )

    if verbose:
        for p in out_paths:
            click.echo(f"  → {p}")
        click.echo(click.style(f"Done. {len(out_paths)} files written.", fg="green"))


@cli.command()
@click.argument("output", type=click.Path())
@click.option(
    "-i",
    "--images",
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help="Image NIfTI files.",
    **_option_kwds,
)
@click.option(
    "-l",
    "--labels",
    multiple=True,
    type=click.Path(exists=True),
    required=True,
    help="Label NIfTI files (same order as --images).",
    **_option_kwds,
)
@click.option(
    "--chunk-shape",
    default="32,32,32",
    help="Chunk shape (comma-separated).",
    **_option_kwds,
)
@click.option("--no-conform", is_flag=True, help="Disable auto-conforming.")
@click.option("-v", "--verbose", is_flag=True, help="Print progress.")
def convert_to_zarr(*, output, images, labels, chunk_shape, no_conform, verbose):
    """Convert NIfTI image+label pairs to a sharded Zarr3 store."""
    from ..datasets.zarr_store import create_zarr_store

    if len(images) != len(labels):
        click.echo(
            click.style(
                f"Error: {len(images)} images but {len(labels)} labels.", fg="red"
            )
        )
        sys.exit(1)

    pairs = list(zip(images, labels))
    chunks = tuple(int(x) for x in chunk_shape.split(","))

    if verbose:
        click.echo(f"Converting {len(pairs)} pairs → {output}")

    store_path = create_zarr_store(
        pairs,
        output,
        chunk_shape=chunks,
        conform=not no_conform,
    )
    click.echo(click.style(f"Zarr store created: {store_path}", fg="green"))


@cli.command()
def merge():
    """Merge multiple models trained with variational weights."""
    click.echo("Not implemented yet.")
    sys.exit(-2)


@cli.command()
@click.argument("outfile")
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint (.ckpt) or weights (.pth).",
    **_option_kwds,
)
@click.option(
    "--model-type",
    default="progressivegan",
    type=click.Choice(["progressivegan", "dcgan"]),
    help="Generative model architecture.",
    **_option_kwds,
)
@click.option(
    "--latent-size",
    type=int,
    default=512,
    help="Latent vector dimension.",
    **_option_kwds,
)
@click.option(
    "--n-samples",
    type=int,
    default=1,
    help="Number of images to generate.",
    **_option_kwds,
)
@click.option(
    "--device",
    default="auto",
    help='Compute device: "auto", "cpu", "cuda", …',
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress messages.", **_option_kwds
)
def generate(
    *,
    outfile,
    model,
    model_type,
    latent_size,
    n_samples,
    device,
    verbose,
):
    """Generate brain volumes from a trained GAN model.

    Saves OUTFILE (NIfTI) for each generated sample.  When ``--n-samples > 1``
    the file stem is suffixed with ``_0``, ``_1``, … before the extension.
    """
    import os

    if device == "auto":
        _device = get_device()
    else:
        _device = torch.device(device)

    if verbose:
        click.echo(f"Using device: {_device}")

    from ..models import get as _get_model

    try:
        factory = _get_model(model_type)
        pt_model = factory(latent_size=latent_size)
        # Support both .ckpt (Lightning) and .pth (state dict)
        if model.endswith(".ckpt"):
            model_cls = type(pt_model)
            pt_model = model_cls.load_from_checkpoint(model, map_location=_device)
        else:
            state = torch.load(model, map_location=_device, weights_only=True)
            pt_model.load_state_dict(state, strict=False)
    except Exception as exc:
        click.echo(click.style(f"ERROR: could not load model: {exc}", fg="red"))
        raise SystemExit(1) from exc

    pt_model = pt_model.to(_device)
    pt_model.eval()

    if verbose:
        click.echo(f"Generating {n_samples} sample(s) …")

    stem, ext = os.path.splitext(outfile)
    if ext == ".gz":
        stem, ext2 = os.path.splitext(stem)
        ext = ext2 + ext

    with torch.no_grad():
        for i in range(n_samples):
            z = torch.randn(1, latent_size, device=_device)
            out = pt_model.generator(z)  # (1, 1, D, H, W)
            arr = out.squeeze().cpu().numpy()
            img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
            path = f"{stem}_{i}{ext}" if n_samples > 1 else outfile
            nib.save(img, path)
            if verbose:
                click.echo(f"  Saved {path}")

    if verbose:
        click.echo(click.style("Done.", fg="green"))


@cli.command()
@click.option(
    "--working-dir",
    required=True,
    type=click.Path(),
    help="Directory with train script and data_manifest.json.",
    **_option_kwds,
)
@click.option(
    "--model-family",
    default="bayesian_vnet",
    help="Model family to use for training.",
    **_option_kwds,
)
@click.option(
    "--max-experiments",
    type=int,
    default=10,
    help="Maximum number of experiments.",
    **_option_kwds,
)
@click.option(
    "--budget-hours",
    type=float,
    default=8.0,
    help="Wall-clock budget in hours.",
    **_option_kwds,
)
@click.option(
    "--budget-minutes",
    type=float,
    default=None,
    help="Wall-clock budget in minutes (overrides --budget-hours).",
    **_option_kwds,
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print per-experiment progress.",
    **_option_kwds,
)
def research(
    *,
    working_dir,
    model_family,
    max_experiments,
    budget_hours,
    budget_minutes,
    verbose,
):
    """Run the autoresearch experiment loop.

    Proposes hyperparameter configs (via Anthropic API or random grid),
    runs training experiments, and keeps improvements.
    Writes ``run_summary.md`` in WORKING_DIR on completion.
    """
    from ..research.loop import run_loop

    if verbose:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    budget_seconds = None
    if budget_minutes is not None:
        budget_seconds = budget_minutes * 60
    results = run_loop(
        working_dir=working_dir,
        model_family=model_family,
        max_experiments=max_experiments,
        budget_hours=budget_hours,
        budget_seconds=budget_seconds,
    )

    # Progress table
    click.echo(
        f"\n{'run_id':>6}  {'val_dice':>10}  {'outcome':<12}  {'failure_reason'}"
    )
    click.echo("-" * 55)
    for r in results:
        dice_str = f"{r.val_dice:.4f}" if r.val_dice is not None else "—"
        click.echo(
            f"{r.run_id:>6}  {dice_str:>10}  {r.outcome:<12}  {r.failure_reason or '—'}"
        )

    summary_path = click.format_filename(f"{working_dir}/run_summary.md")
    click.echo(click.style(f"\nSummary written to {summary_path}", fg="green"))


@cli.command()
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to best_model.pth file.",
    **_option_kwds,
)
@click.option(
    "--config-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to best_config.json file.",
    **_option_kwds,
)
@click.option(
    "--trained-models-path",
    required=True,
    type=click.Path(),
    help="Root of the DataLad-managed trained_models dataset.",
    **_option_kwds,
)
@click.option(
    "--model-family",
    default="bayesian_vnet",
    help="Model family name (used as subdirectory).",
    **_option_kwds,
)
@click.option(
    "--val-dice",
    type=float,
    required=True,
    help="Validation Dice score of the best model.",
    **_option_kwds,
)
@click.option(
    "--source-run-id",
    default="",
    help="Run ID string for traceability.",
    **_option_kwds,
)
def commit(
    *,
    model_path,
    config_path,
    trained_models_path,
    model_family,
    val_dice,
    source_run_id,
):
    """Version the best model with DataLad and push to OSF.

    Copies model weights and config into the trained_models DataLad dataset,
    generates a model card, saves with DataLad, and pushes to OSF.
    """
    from ..research.loop import commit_best_model

    try:
        result = commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_path,
            model_family=model_family,
            val_dice=val_dice,
            source_run_id=source_run_id,
        )
    except ImportError as exc:
        click.echo(click.style(f"ERROR: {exc}", fg="red"))
        raise SystemExit(1) from exc

    click.echo(f"Model versioned at: {result['path']}")
    click.echo(f"DataLad commit: {result['datalad_commit']}")
    if result.get("osf_url"):
        click.echo(click.style(f"OSF URL: {result['osf_url']}", fg="green"))
    else:
        click.echo(click.style("OSF push skipped (no remote configured)", fg="yellow"))


@cli.command()
def save():
    """Save a model to PyTorch format."""
    click.echo("Not implemented yet.")
    sys.exit(-2)


@cli.command()
def evaluate():
    """Evaluate a model's predictions against known labels."""
    click.echo("Not implemented yet.")
    sys.exit(-2)


@cli.command()
def info():
    """Return information about this system."""
    uname = platform.uname()
    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0
    s = f"""\
Python:
 Version: {platform.python_version()}
 Implementation: {platform.python_implementation()}
 64-bit: {sys.maxsize > 2**32}
 Packages:
  Nobrainer: {__version__}
  Nibabel: {nib.__version__}
  Numpy: {np.__version__}
  PyTorch: {torch.__version__}
   CUDA available: {cuda_available}
   CUDA devices: {cuda_devices}

System:
 OSType: {uname.system}
 Release: {uname.release}
 Version: {uname.version}
 Architecture: {uname.machine}

Timestamp: {datetime.datetime.utcnow().strftime('%Y/%m/%d %T')}"""
    click.echo(s)


# ---------------------------------------------------------------------------
# zarr subcommands
# ---------------------------------------------------------------------------


@cli.group()
def zarr():
    """Zarr store management commands."""


@zarr.command("suggest-shards")
@click.option("--n-volumes", required=True, type=int, help="Number of subjects")
@click.option(
    "--volume-shape",
    required=True,
    type=str,
    help="Spatial shape as D,H,W (e.g. 256,256,256)",
)
@click.option("--dtype", default="float32", help="Array dtype")
@click.option(
    "--n-input-files",
    default=None,
    type=int,
    help="Total input files (default: 2×n-volumes)",
)
@click.option("--levels", default=1, type=int, help="Pyramid levels")
def zarr_suggest_shards(n_volumes, volume_shape, dtype, n_input_files, levels):
    """Compute optimal shard parameters for a dataset."""
    import json

    from ..datasets.zarr_store import suggest_shards

    shape = tuple(int(x) for x in volume_shape.split(","))
    result = suggest_shards(
        n_volumes,
        shape,
        dtype=dtype,
        n_input_files=n_input_files,
        levels=levels,
    )
    # Convert tuple to list for JSON serialization
    result["shard_shape"] = list(result["shard_shape"])
    click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# qc subcommands
# ---------------------------------------------------------------------------


@cli.group()
def qc():
    """Quality control tools for brain MRI."""


@qc.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing reference .nii.gz files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Root directory for corrupted outputs.",
)
@click.option(
    "--corruptions",
    default=None,
    help="Comma-separated corruption names (default: all).",
)
@click.option(
    "--severities",
    default="1,2,3,4,5",
    help="Comma-separated severity levels.",
    **_option_kwds,
)
@click.option("--resume/--no-resume", default=True, **_option_kwds)
@click.option("--dry-run", is_flag=True, help="Print plan without writing files.")
def corrupt(*, input_dir, output_dir, corruptions, severities, resume, dry_run):
    """Generate corrupted brain MRI scans for QC benchmarking."""
    from pathlib import Path

    from ..qc.corrupt import generate_corrupted_dataset

    corruption_list = corruptions.split(",") if corruptions else None
    severity_list = [int(s) for s in severities.split(",")]

    metadata = generate_corrupted_dataset(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        corruptions=corruption_list,
        severities=severity_list,
        resume=resume,
        dry_run=dry_run,
    )
    click.echo(f"Generated {len(metadata)} corrupted scans.")


@qc.command()
@click.argument("scan_path", type=click.Path(exists=True))
@click.option(
    "--seg-path",
    default=None,
    type=click.Path(exists=True),
    help="Path to SynthSeg segmentation for CNR/CJV metrics.",
)
def iqms(scan_path, seg_path):
    """Extract image quality metrics from a scan."""
    from ..qc.metrics import extract_iqms

    result = extract_iqms(scan_path, seg_path=seg_path)
    for key, val in result.items():
        click.echo(f"  {key}: {val:.4f}" if val == val else f"  {key}: NaN")


# For debugging only.
if __name__ == "__main__":
    cli()
