import gzip
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pytest
import tensorflow as tf

from nobrainer import io
from nobrainer.tests.utils import csv_of_volumes


def test_read_csv():
    with tempfile.NamedTemporaryFile() as f:
        f.write('foo,bar\nbaz,boo'.encode())
        f.seek(0)
        assert [('foo', 'bar'), ('baz', 'boo')] == io.read_csv(f.name, skip_header=False)

    with tempfile.NamedTemporaryFile() as f:
        f.write('foo,bar\nbaz,boo'.encode())
        f.seek(0)
        assert [('baz', 'boo')] == io.read_csv(f.name, skip_header=True)

    with tempfile.NamedTemporaryFile() as f:
        f.write('foo,bar\nbaz,boo'.encode())
        f.seek(0)
        assert [('baz', 'boo')] == io.read_csv(f.name)

    with tempfile.NamedTemporaryFile() as f:
        f.write('foo|bar\nbaz|boo'.encode())
        f.seek(0)
        assert [('baz', 'boo')] == io.read_csv(f.name, delimiter='|')


def test_read_mapping():
    with tempfile.NamedTemporaryFile() as f:
        f.write('orig,new\n0,1\n20,10\n40,15'.encode())
        f.seek(0)
        assert {0:1, 20:10, 40:15} == io.read_mapping(f.name, skip_header=True)
        # Header is non-integer.
        with pytest.raises(ValueError):
            io.read_mapping(f.name, skip_header=False)

    with tempfile.NamedTemporaryFile() as f:
        f.write('orig,new\n0,1\n20,10\n40'.encode())
        f.seek(0)
        # Last row only has one value.
        with pytest.raises(ValueError):
            io.read_mapping(f.name, skip_header=False)

    with tempfile.NamedTemporaryFile() as f:
        f.write('origFnew\n0F1\n20F10\n40F15'.encode())
        f.seek(0)
        assert {0:1, 20:10, 40:15} == io.read_mapping(f.name, skip_header=True, delimiter='F')


def test_read_volume(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    affine = np.eye(4)

    filename = str(tmp_path / 'foo.nii.gz')
    nib.save(nib.Nifti1Image(data, affine), filename)
    data_loaded = io.read_volume(filename)
    assert np.array_equal(data, data_loaded)

    data_loaded = io.read_volume(filename, dtype=data.dtype)
    assert data.dtype == data_loaded.dtype

    data_loaded, affine_loaded = io.read_volume(filename, return_affine=True)
    assert np.array_equal(data, data_loaded)
    assert np.array_equal(affine, affine_loaded)


def test_convert(csv_of_volumes, tmp_path):
    files = io.read_csv(csv_of_volumes, skip_header=False)
    tfrecords_template = str(tmp_path / 'data-{shard:03d}.tfrecords')
    volumes_per_shard = 12
    io.convert(files, tfrecords_template=tfrecords_template, volumes_per_shard=volumes_per_shard)

    paths = list(tmp_path.glob('data-*.tfrecords'))
    paths = sorted(paths)
    assert len(paths) == 9
    assert (tmp_path / 'data-008.tfrecords').is_file()

    dset = tf.data.TFRecordDataset(list(map(str, paths)), compression_type='GZIP')
    dset = dset.map(io.get_parse_fn(volume_shape=(8, 8, 8), include_affines=True))

    for ref, test in zip(files, dset):
        x, y = ref
        x, x_aff = io.read_volume(x, return_affine=True)
        y, y_aff = io.read_volume(y, return_affine=True)
        assert np.array_equal(x, test[0])
        assert np.array_equal(y, test[1])
        assert np.array_equal(x_aff, test[2])
        assert np.array_equal(y_aff, test[3])

    with pytest.raises(ValueError):
        io.convert(files, tfrecords_template="data/foobar-{}.tfrecords")


def test_verify_features_labels(csv_of_volumes):
    files = io.read_csv(csv_of_volumes, skip_header=False)
    io.verify_features_labels(files, volume_shape=(8, 8, 8))


def test_is_gzipped(tmp_path):
    filename = str(tmp_path / 'test.gz')
    with gzip.GzipFile(filename, 'w') as f:
        f.write("i'm more than a test!".encode())
    assert io._is_gzipped(filename)

    with open(filename, 'w') as f:
        f.write("i'm just a test...")
    assert not io._is_gzipped(filename)
