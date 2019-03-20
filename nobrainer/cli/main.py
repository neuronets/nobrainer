"""Main command-line interface for nobrainer."""

import inspect
import json
import logging
import os
import pprint
import sys

import click
import nibabel as nib
import numpy as np
import skimage
import tensorflow as tf

from nobrainer import __version__
from nobrainer.io import convert as _convert
from nobrainer.io import verify_features_labels as _verify_features_labels
from nobrainer.io import read_csv as _read_csv
from nobrainer.io import read_volume as _read_volume
from nobrainer.losses import get as _get_loss
from nobrainer.training import train as _train
from nobrainer.utils import _get_all_cpus
from nobrainer.volume import from_blocks_numpy as _from_blocks_numpy
from nobrainer.volume import get_dataset as _get_dataset
from nobrainer.volume import get_steps_per_epoch as _get_steps_per_epoch
from nobrainer.volume import standardize_numpy as _standardize_numpy
from nobrainer.volume import to_blocks_numpy as _to_blocks_numpy


_option_kwds = {
    'show_default': True
}


class JSONParamType(click.ParamType):
    name = 'json'

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            self.fail('%s is not valid JSON' % value, param, ctx)


@click.group()
@click.version_option(__version__, message='%(prog)s version %(version)s')
def cli():
    """A framework for developing neural network models for 3D image processing."""
    return


@cli.command()
@click.option('-c', '--csv', type=click.Path(exists=True), required=True, **_option_kwds)
@click.option('-t', '--tfrecords-template', default='tfrecords/data_shard-{shard:03d}.tfrecords', required=True, **_option_kwds)
@click.option('-s', '--volume-shape', nargs=3, type=int, required=True, **_option_kwds)
@click.option('-n', '--volumes-per-shard', type=int, default=100, help='Number of volume pairs per TFRecords file.', **_option_kwds)
@click.option('--to-ras/--no-to-ras', default=True, help='Reorient volumes to RAS before saving to TFRecords.', **_option_kwds)
@click.option('--gzip/--no-gzip', default=True, help='Compress TFRecords with gzip (highly recommended).', **_option_kwds)
@click.option('--verify-volumes/--no-verify-volumes', default=True, help='Verify volume pairs before converting. This option is highly recommended, as it checks that shapes of features and labels are equal to "volume-shape", that labels are (or can safely be coerced to) an integer type, and that labels are all >= 0.', **_option_kwds)
@click.option('-j', '--num-parallel-calls', default=-1, type=int, help='Number of processes to use. If -1, uses all available processes.', **_option_kwds)
@click.option('-v', '--verbose', is_flag=True, help='Print progress bar.', **_option_kwds)
def convert(*, csv, tfrecords_template, volume_shape, volumes_per_shard, to_ras, gzip, verify_volumes, num_parallel_calls, verbose):
    """Convert medical imaging volumes to TFRecords.

    Volumes must all be the same shape. This will overwrite existing TFRecords files.
    """
    # TODO: improve docs.
    volume_filepaths = _read_csv(csv)
    num_parallel_calls = None if num_parallel_calls == -1 else num_parallel_calls
    if num_parallel_calls is None:
        num_parallel_calls = _get_all_cpus()

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
            verbose=1)

        if not invalid_pairs:
            click.echo(click.style('Passed verification.', fg='green'))
        else:
            click.echo(click.style('Failed verification.', fg='red'))
            click.echo("Found {} invalid pairs of volumes. These files might not all have shape {}, the labels might not be an integer type or coercible to integer type, or the labels might not be >= 0.".format(len(invalid_pairs), volume_shape))
            for pair in invalid_pairs:
                click.echo(pair[0])
                click.echo(pair[1])
            sys.exit(-1)

    _convert(
        volume_filepaths=volume_filepaths,
        tfrecords_template=tfrecords_template,
        volumes_per_shard=volumes_per_shard,
        to_ras=to_ras,
        gzip_compressed=gzip,
        num_parallel_calls=num_parallel_calls,
        verbose=verbose)

    click.echo(click.style('Finished conversion to TFRecords.', fg='green'))


@cli.command()
def merge():
    """Merge multiple models trained with variational weights.

    These models must have the same architecture and should have been trained
    from the same initial model.
    """
    click.echo("Not implemented yet. In the future, this command will be used for merging models.")
    sys.exit(-2)


@cli.command()
@click.argument('infile')
@click.argument('outfile')
@click.option('-m', '--model', type=click.Path(exists=True), required=True, help='Path to model HDF5 file.', **_option_kwds)
@click.option('-b', '--block-shape', default=(128, 128, 128), type=int, nargs=3, help='Shape of sub-volumes on which to predict.', **_option_kwds)
@click.option('-v', '--verbose', is_flag=True, help='Print progress bar.', **_option_kwds)
def predict(*, infile, outfile, model, block_shape, verbose):
    """Predict labels from features using a trained model.

    The predictions are saved to OUTFILE.
    """

    if not verbose:
        # Supress most logging messages.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel(logging.ERROR)

    if os.path.exists(outfile):
        raise FileExistsError(
            "Output file already exists. Will not overwrite {}".format(outfile))

    x, affine = _read_volume(infile, dtype=np.float32, return_affine=True)
    if x.ndim != 3:
        raise ValueError("Input volume must be rank 3, got rank {}".format(x.ndim))
    original_shape = x.shape
    # TODO: make this more general. For now, all of our models are trained on
    # blocks with dimensions 128x128x128, which are taken easily from a volume
    # with dimensions 256x256x256.
    required_shape = (256, 256, 256)
    must_resize = False
    if x.shape != required_shape:
        must_resize = True
        if verbose:
            click.echo("Resizing volume from shape {} to shape {}".format(x.shape, required_shape))
        x = skimage.transform.resize(
            x,
            output_shape=required_shape,
            order=1,  # linear
            mode='constant',
            preserve_range=True,
            anti_aliasing=False)

    x = _standardize_numpy(x)
    x_blocks = _to_blocks_numpy(x, block_shape=block_shape)
    x_blocks = x_blocks[..., None]  # Add grayscale channel.

    model = tf.keras.models.load_model(model, compile=False)
    if verbose:
        click.echo("Predicting ...")
    y_blocks = model.predict(x_blocks, batch_size=1, verbose=verbose)

    # Binarize segmentation.
    y_blocks = y_blocks > 0.3

    # Collapse the last dimension, depending on number of output classes.
    if y_blocks.shape[-1] == 1:
        y_blocks = y_blocks.squeeze(-1)
    else:
        y_blocks = y_blocks.argmax(-1)

    y = _from_blocks_numpy(y_blocks, x.shape)

    if must_resize:
        if verbose:
            click.echo("Resizing volume from shape {} to shape {}".format(y.shape, original_shape))
        y = skimage.transform.resize(
            y,
            output_shape=original_shape,
            order=0,  # nearest neighbor
            mode='constant',
            preserve_range=True,
            anti_aliasing=False)

    img = nib.spatialimages.SpatialImage(y.astype(np.int32), affine=affine)
    nib.save(img, outfile)
    if verbose:
        click.echo("Output saved to {}".format(outfile))


@cli.command()
def save():
    """Save a model to SavedModel type."""
    click.echo("Not implemented yet. In the future, this command will be used for saving.")
    sys.exit(-2)


@cli.command()
@click.option('-m', '--model', required=True, help='Neural network model architecture.', **_option_kwds)
@click.option('-t', '--tfrecords-pattern', required=True, **_option_kwds)
@click.option('-c', '--n-classes', type=int, required=True, **_option_kwds)
@click.option('-b', '--batch-size', type=int, required=True, **_option_kwds)
@click.option('-s', '--volume-shape', nargs=3, type=int, required=True, **_option_kwds)
@click.option('--block-shape', nargs=3, type=int, help='Shape of non-overlapping blocks to take from full volumes. Default behavior is to use full volumes.', **_option_kwds)
@click.option('-e', '--n-epochs', type=int, **_option_kwds)
@click.option('-i', '--initial-epoch', type=int, default=0, **_option_kwds)
@click.option('-n', '--n-volumes', type=int, required=True, help='Number of full volumes contained in the TFRecords files.', **_option_kwds)
@click.option('-l', '--loss', required=True, help='Loss function to use.', **_option_kwds)
@click.option('-lr', '--learning-rate', type=float, required=True, help='Learning rate for Adam optimizer.', **_option_kwds)
@click.option('--shuffle-buffer-size', default=20, type=int, help='Size of shuffle buffer (i.e., number of volumes to shuffle at a time).', **_option_kwds)
@click.option('--mapping', type=click.Path(exists=True), help='Path to CSV that contains replacement mapping for values in labels.', **_option_kwds)
@click.option('--model-kwds', type=JSONParamType(), help='Keywords to pass to model, in JSON format.', **_option_kwds)
@click.option('--augment/--no-augment', default=False, help='Apply random rigid transformations to training data. This will slow down training.', **_option_kwds)
@click.option('--multi-gpu/--no-multi-gpu', default=False, help='Train on multiple GPUs on the same machine.', **_option_kwds)
@click.option('-j', '--num-parallel-calls', default=-1, type=int, help='Number of processes to use for data processing. If -1, uses all available processes.', **_option_kwds)
def train(*,
          model,
          tfrecords_pattern,
          n_classes,
          batch_size,
          volume_shape,
          block_shape,
          n_epochs,
          initial_epoch,
          n_volumes,
          loss,
          learning_rate,
          shuffle_buffer_size,
          mapping,
          augment,
          model_kwds,
          multi_gpu,
          num_parallel_calls):
    """Train a model."""

    dataset = _get_dataset(
        file_pattern=tfrecords_pattern,
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        block_shape=block_shape,
        n_epochs=n_epochs,
        mapping=mapping,
        augment=augment,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_calls=num_parallel_calls)

    steps_per_epoch = _get_steps_per_epoch(
        n_volumes=n_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size)

    # TODO: validation dataset

    if multi_gpu:
        # As of March 05, 2019, keras optimizers only have experimental support
        # with MirroredStrategy training.
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Searches custom nobrainer losses as well as standard tf.keras losses.
    loss = _get_loss(loss)
    # Instantiate loss object if it is a class.
    if inspect.isclass(loss):
        loss = loss()


    if model_kwds is None:
        model_kwds = {}

    click.echo('Beginning to train model in directory {}'.format(os.getcwd()))
    click.echo('Parameters:')
    pprint.pprint({
        'model': model,
        'tfrecords_pattern': tfrecords_pattern,
        'n_classes': n_classes,
        'batch_size': batch_size,
        'volume_shape': volume_shape,
        'block_shape': block_shape,
        'n_epochs': n_epochs,
        'initial_epoch': initial_epoch,
        'n_volumes': n_volumes,
        'loss': loss,
        'learning_rate': learning_rate,
        'shuffle_buffer_size': shuffle_buffer_size,
        'mapping': mapping,
        'augment': augment,
        'model_kwds': model_kwds,
        'multi_gpu': multi_gpu,
        'num_parallel_calls': num_parallel_calls})

    history = _train(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        loss=loss,
        steps_per_epoch=steps_per_epoch,
        model_kwds=model_kwds,
        n_epochs=n_epochs,
        initial_epoch=initial_epoch,
        # Use default metrics and callbacks.
        metrics=None,
        callbacks=None,
        multi_gpu=multi_gpu,
        devices=None)

    click.echo(click.style('Finished training model.', fg='green'))


@cli.command()
def evaluate():
    """Evaluate a model's predictions against known labels."""
    click.echo("Not implemented yet. In the future, this command will be used for evaluation.")
    sys.exit(-2)


# For debugging only.
if __name__ == '__main__':
    cli()
