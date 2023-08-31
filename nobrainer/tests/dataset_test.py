import glob
import os.path as op
import shutil
import tempfile

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from .. import dataset, intensity_transforms, io, spatial_transforms, tfrecord, utils


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
    dset = dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern,
        n_volumes=10,
        volume_shape=volume_shape,
        scalar_labels=True,
        n_classes=1,
        num_parallel_calls=num_parallel_calls,
    ).batch(batch_size)

    y_orig = np.array([y for _, y in filepaths])
    y_from_dset = (
        np.concatenate([y for _, y in dset.dataset.as_numpy_iterator()])
        .flatten()
        .astype(int)
    )
    assert_array_equal(y_orig, y_from_dset)
    shutil.rmtree(temp_dir)


def create_dummy_niftis(shape, n_volumes, outdir):
    array_data = np.zeros(shape, dtype=np.int16)
    affine = np.diag([1, 2, 3, 1])
    array_img = nib.Nifti1Image(array_data, affine)
    for volume in range(n_volumes):
        nib.save(array_img, op.join(outdir, f"image_{volume}.nii.gz"))

    return glob.glob(op.join(outdir, "image_*.nii.gz"))


def test_get_dataset_errors():
    temp_dir = tempfile.mkdtemp()
    file_pattern = op.join(temp_dir, "does_not_exist-*.tfrec")
    with pytest.raises(ValueError):
        dataset.Dataset.from_tfrecords(
            file_pattern,
            None,
            (256, 256, 256),
            n_classes=1,
        )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("examples_per_shard", [1, 3])
@pytest.mark.parametrize("volume_shape", [(64, 64, 64), (64, 64, 64, 3)])
@pytest.mark.parametrize("num_parallel_calls", [1, 2])
def test_get_dataset_shapes(
    volume_shape, examples_per_shard, batch_size, num_parallel_calls
):
    temp_dir = tempfile.mkdtemp()
    nifti_paths = create_dummy_niftis(volume_shape, 10, temp_dir)
    filepaths = [(x, i) for i, x in enumerate(nifti_paths)]
    file_pattern = write_tfrecs(
        filepaths, temp_dir, examples_per_shard=examples_per_shard
    )
    dset = dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern,
        n_volumes=len(filepaths),
        volume_shape=volume_shape,
        scalar_labels=True,
        n_classes=1,
        num_parallel_calls=num_parallel_calls,
    ).batch(batch_size)

    output_volume_shape = volume_shape if len(volume_shape) > 3 else volume_shape + (1,)
    output_volume_shape = (batch_size,) + output_volume_shape
    shapes = [x.shape for x, _ in dset.dataset.as_numpy_iterator()]
    assert all([_shape == output_volume_shape for _shape in shapes])
    shutil.rmtree(temp_dir)


def test_get_dataset_errors_augmentation():
    temp_dir = tempfile.mkdtemp()
    file_pattern = op.join(temp_dir, "does_not_exist-*.tfrec")
    with pytest.raises(ValueError):
        dataset.Dataset.from_tfrecords(
            file_pattern=file_pattern,
            n_volumes=10,
            volume_shape=(256, 256, 256),
            n_classes=1,
        ).augment = [
            (
                intensity_transforms.addGaussianNoise,
                {"noise_mean": 0.1, "noise_std": 0.5},
            ),
            (spatial_transforms.randomflip_leftright),
        ]
    shutil.rmtree(temp_dir)


# TODO: need to implement this soon.
@pytest.mark.xfail
def test_get_dataset():
    assert False


def test_get_steps_per_epoch():
    volume_shape = (256, 256, 256)
    temp_dir = tempfile.mkdtemp()
    nifti_paths = create_dummy_niftis(volume_shape, 10, temp_dir)
    filepaths = [(x, i) for i, x in enumerate(nifti_paths)]
    file_pattern = write_tfrecs(filepaths, temp_dir, examples_per_shard=1)
    dset = dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern.replace("*", "000"),
        n_volumes=1,
        volume_shape=volume_shape,
        block_shape=(64, 64, 64),
        scalar_labels=True,
        n_classes=1,
    )
    assert dset.get_steps_per_epoch() == 64

    dset = dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern.replace("*", "000"),
        n_volumes=1,
        volume_shape=volume_shape,
        block_shape=(64, 64, 64),
        scalar_labels=True,
        n_classes=1,
    ).batch(64)
    assert dset.get_steps_per_epoch() == 1

    dset = dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern.replace("*", "000"),
        n_volumes=1,
        volume_shape=volume_shape,
        block_shape=(64, 64, 64),
        scalar_labels=True,
        n_classes=1,
    ).batch(63)
    assert dset.get_steps_per_epoch() == 2

    dset = dataset.Dataset.from_tfrecords(
        file_pattern=file_pattern,
        n_volumes=10,
        volume_shape=volume_shape,
        block_shape=(128, 128, 128),
        scalar_labels=True,
        n_classes=1,
    ).batch(4)
    assert dset.get_steps_per_epoch() == 20

    shutil.rmtree(temp_dir)
