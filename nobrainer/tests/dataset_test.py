import os.path as op
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from .. import dataset, io, tfrecord, utils


@pytest.fixture(scope="session")
def tmp_data_filepaths():
    temp_dir = tempfile.mkdtemp()
    csv_of_filepaths = utils.get_data(cache_dir=temp_dir)
    filepaths = io.read_csv(csv_of_filepaths)
    yield filepaths
    shutil.rmtree(temp_dir)


def write_tfrecs(filepaths, outdir, examples_per_shard):
    tfrecord.write(
        features_labels=filepaths,
        filename_template=op.join(outdir, "data_shard-{shard:03d}.tfrec"),
        examples_per_shard=examples_per_shard,
    )
    return op.join(outdir, "data_shard-*.tfrec")


@pytest.mark.parametrize("examples_per_shard", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_parallel_calls", [1, 2])
def test_get_dataset_maintains_order(
    tmp_data_filepaths, examples_per_shard, batch_size, num_parallel_calls
):
    filepaths = [(x, i) for i, (x, _) in enumerate(tmp_data_filepaths)]
    temp_dir = tempfile.mkdtemp()
    file_pattern = write_tfrecs(
        filepaths, temp_dir, examples_per_shard=examples_per_shard
    )
    volume_shape = (256, 256, 256)
    dset = dataset.get_dataset(
        file_pattern=file_pattern,
        n_classes=1,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        n_epochs=1,
        num_parallel_calls=num_parallel_calls,
    )
    y_orig = np.array([y for _, y in filepaths])
    y_from_dset = (
        np.concatenate([y for _, y in dset.as_numpy_iterator()]).flatten().astype(int)
    )
    assert_array_equal(y_orig, y_from_dset)
    shutil.rmtree(temp_dir)


# TODO: need to implement this soon.
@pytest.mark.xfail
def test_get_dataset():
    assert False


def test_get_steps_per_epoch():
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=1,
        volume_shape=(256, 256, 256),
        block_shape=(64, 64, 64),
        batch_size=1,
    )
    assert nsteps == 64
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=1,
        volume_shape=(256, 256, 256),
        block_shape=(64, 64, 64),
        batch_size=64,
    )
    assert nsteps == 1
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=1,
        volume_shape=(256, 256, 256),
        block_shape=(64, 64, 64),
        batch_size=63,
    )
    assert nsteps == 2
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=10,
        volume_shape=(256, 256, 256),
        block_shape=(128, 128, 128),
        batch_size=4,
    )
    assert nsteps == 20
