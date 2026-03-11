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
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def merge():
    """Merge multiple models trained with variational weights."""
    click.echo("Not implemented yet.")
    sys.exit(-2)


@cli.command()
def generate():
    """Generate images from latents using a trained GAN model (Phase 5)."""
    click.echo("Not implemented yet. Generative models are in Phase 5 (US3).")
    sys.exit(-2)


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


# For debugging only.
if __name__ == "__main__":
    cli()
