"""Main command-line interface for nobrainer."""

import datetime
import glob
import json
import logging
import os
import platform
import sys

import click
import nibabel as nib
import numpy as np
import skimage.measure
import skimage.transform
import tensorflow as tf
import tensorflow_probability as tfp

from .. import __version__
from ..io import read_csv as _read_csv
from ..io import read_volume as _read_volume
from ..io import verify_features_labels as _verify_features_labels
from ..prediction import _transform_and_predict
from ..tfrecord import write as _write_tfrecord
from ..utils import get_num_parallel
from ..volume import from_blocks_numpy as _from_blocks_numpy
from ..volume import standardize_numpy as _standardize_numpy
from ..volume import to_blocks_numpy as _to_blocks_numpy

_option_kwds = {"show_default": True}


class JSONParamType(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            self.fail("%s is not valid JSON" % value, param, ctx)


@click.group()
@click.version_option(__version__, message="%(prog)s version %(version)s")
def cli():
    """A framework for developing neural network models for 3D image processing."""
    return


@cli.command()
@click.option(
    "-c", "--csv", type=click.Path(exists=True), required=True, **_option_kwds
)
@click.option(
    "-t",
    "--tfrecords-template",
    default="tfrecords/data_shard-{shard:03d}.tfrec",
    required=True,
    **_option_kwds,
)
@click.option("-s", "--volume-shape", nargs=3, type=int, required=True, **_option_kwds)
@click.option(
    "-n",
    "--examples-per-shard",
    type=int,
    default=100,
    help="Number of (feature, label) pairs per TFRecord file.",
    **_option_kwds,
)
@click.option(
    "--to-ras/--no-to-ras",
    default=True,
    help="Reorient volumes to RAS before saving to TFRecords.",
    **_option_kwds,
)
@click.option(
    "--gzip/--no-gzip",
    default=True,
    help="Compress TFRecords with gzip (highly recommended).",
    **_option_kwds,
)
@click.option(
    "--verify-volumes/--no-verify-volumes",
    default=True,
    help=(
        "Verify volume pairs before converting. This option is highly recommended, as"
        ' it checks that shapes of features and labels are equal to "volume-shape",'
        " that labels are (or can safely be coerced to) an integer type, and that"
        " labels are all >= 0."
    ),
    **_option_kwds,
)
@click.option(
    "-j",
    "--num-parallel-calls",
    default=-1,
    type=int,
    help="Number of processes to use. If -1, uses all available processes.",
    **_option_kwds,
)
@click.option(
    "--multi-resolution/--no-multi-resolution",
    default=False,
    help="Create tfrecords for multiple resolutions. if set, labels in csv need to be scalar.",
    **_option_kwds,
)
@click.option(
    "--start-resolution",
    type=int,
    default=4,
    help=(
        "Set if `multi-resolution` is true. Indicates smallest resolution for "
        "tfrecords, all resolutions in exponents of 2 from `start-resolution` to"
        " `volume-shape` are generated as tfrecords"
    ),
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress bar.", **_option_kwds
)
def convert(
    *,
    csv,
    tfrecords_template,
    volume_shape,
    examples_per_shard,
    to_ras,
    gzip,
    verify_volumes,
    num_parallel_calls,
    multi_resolution,
    start_resolution,
    verbose,
):
    """Convert medical imaging volumes to TFRecords.

    Volumes must all be the same shape. This will overwrite existing TFRecord files.

    Labels can be volumetric or scalar.
    """
    # TODO: improve docs.
    volume_filepaths = _read_csv(csv)
    num_parallel_calls = (
        get_num_parallel() if num_parallel_calls == -1 else num_parallel_calls
    )

    _dirname = os.path.dirname(tfrecords_template)
    if not os.path.exists(_dirname):
        raise ValueError("directory does not exist: {}".format(_dirname))

    if verify_volumes:
        invalid_pairs = _verify_features_labels(
            volume_filepaths=volume_filepaths,
            volume_shape=volume_shape,
            check_shape=True,
            check_labels_int=True,
            check_labels_gte_zero=True,
            num_parallel_calls=None,
            verbose=1,
        )

        if not invalid_pairs:
            click.echo(click.style("Passed verification.", fg="green"))
        else:
            click.echo(click.style("Failed verification.", fg="red"))
            click.echo(
                f"Found {len(invalid_pairs)} invalid pairs of volumes. These files"
                " might not all have shape {volume_shape}, the labels might not be an"
                " integer type or coercible to integer type, or the labels might not"
                " be >= 0."
            )
            for pair in invalid_pairs:
                click.echo(pair[0])
                click.echo(pair[1])
            sys.exit(-1)

    if multi_resolution:
        start_resolution_log = np.log2(start_resolution).astype(np.int32)
        target_resolution_log = np.log2(volume_shape[0]).astype(np.int32)
        resolutions = [
            2 ** res for res in range(start_resolution_log, target_resolution_log + 1)
        ]
    else:
        resolutions = None

    _write_tfrecord(
        features_labels=volume_filepaths,
        filename_template=tfrecords_template,
        examples_per_shard=examples_per_shard,
        to_ras=to_ras,
        compressed=gzip,
        processes=num_parallel_calls,
        multi_resolution=multi_resolution,
        resolutions=resolutions,
        verbose=verbose,
    )

    click.echo(click.style("Finished conversion to TFRecords.", fg="green"))


@cli.command()
def merge():
    """Merge multiple models trained with variational weights.

    These models must have the same architecture and should have been trained
    from the same initial model.
    """
    click.echo(
        "Not implemented yet. In the future, this command will be used for merging"
        " models."
    )
    sys.exit(-2)


@cli.command()
@click.argument("infile")
@click.argument("outfile")
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model HDF5 file.",
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
    "-r",
    "--resize-features-to",
    default=(256, 256, 256),
    type=int,
    nargs=3,
    help="Resize features to this size before taking blocks and predicting.",
    **_option_kwds,
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.3,
    help=(
        "Threshold used to binarize model output. Only used in binary prediction and"
        " must be in (0, 1)."
    ),
    **_option_kwds,
)
@click.option(
    "-l",
    "--largest-label",
    is_flag=True,
    help=(
        "Zero out all values not connected to the largest contiguous label (not"
        " including 0 values). This remove false positives in binary prediction."
    ),
    **_option_kwds,
)
@click.option(
    "--rotate-and-predict",
    is_flag=True,
    help=(
        "Average the prediction with a prediction on a rotated (and subsequently"
        " un-rotated) volume. This can produce a better overall prediction."
    ),
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress bar.", **_option_kwds
)
def predict(
    *,
    infile,
    outfile,
    model,
    block_shape,
    resize_features_to,
    threshold,
    largest_label,
    rotate_and_predict,
    verbose,
):
    """Predict labels from features using a trained model.

    The predictions are saved to OUTFILE.
    """

    if not verbose:
        # Suppress most logging messages.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel(logging.ERROR)

    if os.path.exists(outfile):
        raise FileExistsError(
            "Output file already exists. Will not overwrite {}".format(outfile)
        )

    x, affine = _read_volume(infile, dtype=np.float32, return_affine=True)
    if x.ndim != 3:
        raise ValueError("Input volume must be rank 3, got rank {}".format(x.ndim))
    original_shape = x.shape
    required_shape = resize_features_to
    must_resize = False
    if x.shape != required_shape:
        must_resize = True
        if verbose:
            click.echo(
                "Resizing volume from shape {} to shape {}".format(
                    x.shape, required_shape
                )
            )
        x = skimage.transform.resize(
            x,
            output_shape=required_shape,
            order=1,  # linear
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        )

    x = _standardize_numpy(x)
    x_blocks = _to_blocks_numpy(x, block_shape=block_shape)
    x_blocks = x_blocks[..., None]  # Add grayscale channel.

    model = tf.keras.models.load_model(model, compile=False)
    if verbose:
        click.echo("Predicting ...")
    try:
        y_blocks = model.predict(x_blocks, batch_size=1, verbose=verbose)
    except Exception:
        click.echo(click.style("ERROR: prediction failed. See error trace.", fg="red"))
        raise

    # Collapse the last dimension, depending on number of output classes.
    is_binary_prediction = y_blocks.shape[-1] == 1
    if is_binary_prediction:
        y_blocks = y_blocks.squeeze(-1)
    else:
        y_blocks = y_blocks.argmax(-1)

    y = _from_blocks_numpy(y_blocks, x.shape)

    # Rotate the volume, predict, undo the rotation, and average with original
    # prediction.
    if rotate_and_predict:
        if not is_binary_prediction:
            raise ValueError("Cannot transform and predict on multi-class output.")
        if verbose:
            click.echo("Predicting on rotated volume ...")
        y_other = _transform_and_predict(
            model=model,
            x=x,
            block_shape=block_shape,
            rotation=[np.pi / 4, np.pi / 4, 0],
            translation=[0, 0, 0],
            verbose=verbose,
        )
        if verbose:
            click.echo("Averaging predictions ...")
        y = np.mean([y, y_other], axis=0)

    if is_binary_prediction:
        if threshold <= 0 or threshold >= 1:
            raise ValueError("Threshold must be in (0, 1).")
        y = y > threshold

    if must_resize:
        if verbose:
            click.echo(
                "Resizing volume from shape {} to shape {}".format(
                    y.shape, original_shape
                )
            )
        y = skimage.transform.resize(
            y,
            output_shape=original_shape,
            order=0,  # nearest neighbor
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        )

    if largest_label:
        if not is_binary_prediction:
            raise ValueError(
                "Removing all labels except the largest is only allowed with binary"
                " prediction."
            )
        if verbose:
            click.echo("Removing all labels except largest ...")
        labels, n_labels = skimage.measure.label(y, return_num=True)
        # Do not consider 0 values.
        d = {(labels == label).sum(): label for label in range(1, n_labels + 1)}
        largest_label = d[max(d.keys())]
        if verbose:
            click.echo(
                "Zeroed {} region(s) not contiguous with largest label.".format(
                    n_labels - 2
                )
            )
        y = (labels == largest_label).astype(np.int32)

    img = nib.spatialimages.SpatialImage(y.astype(np.int32), affine=affine)
    nib.save(img, outfile)
    if verbose:
        click.echo("Output saved to {}".format(outfile))


@cli.command()
@click.argument("outfile")
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to saved models directory containing the HDF5 files.",
    **_option_kwds,
)
@click.option(
    "-l",
    "--latent-size",
    type=int,
    default=1024,
    help=("Input latent size for the generator."),
    **_option_kwds,
)
@click.option(
    "--drange-in",
    default=(-1, 1),
    type=int,
    nargs=2,
    help="Range of values of image generated by model.",
    **_option_kwds,
)
@click.option(
    "--drange-out",
    default=(0, 255),
    type=int,
    nargs=2,
    help="Desired output range of values of image.",
    **_option_kwds,
)
@click.option(
    "-o",
    "--output-shape",
    default=(128, 128, 128),
    type=int,
    nargs=3,
    help="Shape of sub-volumes to generate.",
    **_option_kwds,
)
@click.option(
    "--multi-resolution",
    is_flag=True,
    help=(
        "Generate all resolutions available from the progressive generator using the same latents."
    ),
    **_option_kwds,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Print progress bar.", **_option_kwds
)
def generate(
    *,
    outfile,
    model,
    latent_size,
    drange_in,
    drange_out,
    output_shape,
    multi_resolution,
    verbose,
):
    """Generate images from latents using a trained GAN model.

    The generated image is saved to OUTFILE.
    """

    if not verbose:
        # Suppress most logging messages.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel(logging.ERROR)

    if os.path.exists(outfile):
        raise FileExistsError(
            "Output file already exists. Will not overwrite {}".format(outfile)
        )

    if verbose:
        click.echo("Generating ...")
    try:
        latents = tf.random.normal((1, latent_size))

        if multi_resolution:
            images = []
            model_paths = glob.glob(os.path.join(model, "generator_res*"))
            resolutions = []

            for model_path in model_paths:
                generator = tf.saved_model.load(model_path)
                generate = generator.signatures["serving_default"]
                img = generate(latents)["generated"]
                img = np.squeeze(img)
                images.append(img)
                resolutions.append(os.path.splitext(model_path)[0].split("_")[-1])

        else:
            output_resolution = int(output_shape[0])
            model = os.path.join(model, "generator_res_{}".format(output_resolution))
            generator = tf.saved_model.load(model)
            generate = generator.signatures["serving_default"]
            img = generate(latents)["generated"]
            img = np.squeeze(img)

    except Exception:
        click.echo(click.style("ERROR: generation failed. See error trace.", fg="red"))
        raise

    if verbose:
        click.echo("Saving ...")
    if multi_resolution:
        for img, resolution in zip(images, resolutions):
            img = nib.Nifti1Image(img.astype(np.uint8), np.eye(4))
            # Add resolution to the outfile as an id
            nib.save(
                img,
                "{0}_res_{3}.{1}.{2}".format(*outfile.split(".") + [int(resolution)]),
            )
    else:
        img = nib.Nifti1Image(img.astype(np.uint8), np.eye(4))
        nib.save(img, outfile)

    if verbose:
        click.echo("Output saved to {}".format(outfile))


@cli.command()
def save():
    """Save a model to SavedModel type."""
    click.echo(
        "Not implemented yet. In the future, this command will be used for saving."
    )
    sys.exit(-2)


@cli.command()
def evaluate():
    """Evaluate a model's predictions against known labels."""
    click.echo(
        "Not implemented yet. In the future, this command will be used for evaluation."
    )
    sys.exit(-2)


@cli.command()
def info():
    """Return information about this system."""
    uname = platform.uname()
    s = f"""\
Python:
 Version: {platform.python_version()}
 Implementation: {platform.python_implementation()}
 64-bit: {sys.maxsize > 2**32}
 Packages:
  Nobrainer: {__version__}
  Nibabel: {nib.__version__}
  Numpy: {np.__version__}
  TensorFlow: {tf.__version__}
   GPU support: {tf.test.is_built_with_gpu_support()}
   GPU available: {bool(tf.config.list_physical_devices('GPU'))}
  TensorFlow-Probability: {tfp.__version__}

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
