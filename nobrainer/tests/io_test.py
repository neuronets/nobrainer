import gzip
import tempfile

import nibabel as nib
import numpy as np
import pytest

from .utils import csv_of_volumes  # noqa: F401
from .. import io


def test_read_csv():
    with tempfile.NamedTemporaryFile() as f:
        f.write("foo,bar\nbaz,boo".encode())
        f.seek(0)
        assert [("foo", "bar"), ("baz", "boo")] == io.read_csv(
            f.name, skip_header=False
        )

    with tempfile.NamedTemporaryFile() as f:
        f.write("foo,bar\nbaz,boo".encode())
        f.seek(0)
        assert [("baz", "boo")] == io.read_csv(f.name, skip_header=True)

    with tempfile.NamedTemporaryFile() as f:
        f.write("foo,bar\nbaz,boo".encode())
        f.seek(0)
        assert [("baz", "boo")] == io.read_csv(f.name)

    with tempfile.NamedTemporaryFile() as f:
        f.write("foo|bar\nbaz|boo".encode())
        f.seek(0)
        assert [("baz", "boo")] == io.read_csv(f.name, delimiter="|")


def test_read_mapping():
    with tempfile.NamedTemporaryFile() as f:
        f.write("orig,new\n0,1\n20,10\n40,15".encode())
        f.seek(0)
        assert {0: 1, 20: 10, 40: 15} == io.read_mapping(f.name, skip_header=True)
        # Header is non-integer.
        with pytest.raises(ValueError):
            io.read_mapping(f.name, skip_header=False)

    with tempfile.NamedTemporaryFile() as f:
        f.write("orig,new\n0,1\n20,10\n40".encode())
        f.seek(0)
        # Last row only has one value.
        with pytest.raises(ValueError):
            io.read_mapping(f.name, skip_header=False)

    with tempfile.NamedTemporaryFile() as f:
        f.write("origFnew\n0F1\n20F10\n40F15".encode())
        f.seek(0)
        assert {0: 1, 20: 10, 40: 15} == io.read_mapping(
            f.name, skip_header=True, delimiter="F"
        )


def test_read_volume(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    affine = np.eye(4)

    filename = str(tmp_path / "foo.nii.gz")
    nib.save(nib.Nifti1Image(data, affine), filename)
    data_loaded = io.read_volume(filename)
    assert np.array_equal(data, data_loaded)

    data_loaded = io.read_volume(filename, dtype=data.dtype)
    assert data.dtype == data_loaded.dtype

    data_loaded, affine_loaded = io.read_volume(filename, return_affine=True)
    assert np.array_equal(data, data_loaded)
    assert np.array_equal(affine, affine_loaded)

    data = np.random.rand(8, 8, 8).astype(np.float32)
    affine = np.array([[1.5, 0, 1.2, 0], [0.8, 0.8, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    filename = str(tmp_path / "foo_asr.nii.gz")
    nib.save(nib.Nifti1Image(data, affine), filename)
    data_loaded = io.read_volume(filename, to_ras=True)
    assert not np.array_equal(data, data_loaded)
    data_loaded = io.read_volume(filename, to_ras=False)
    assert np.array_equal(data, data_loaded)


def test_verify_features_nonscalar_labels(csv_of_volumes):  # noqa: F811
    files = io.read_csv(csv_of_volumes, skip_header=False)
    invalid = io.verify_features_labels(
        files, volume_shape=(8, 8, 8), num_parallel_calls=1
    )
    assert not invalid
    # TODO: add more cases.


def test_verify_features_scalar_labels(csv_of_volumes):  # noqa: F811
    files = io.read_csv(csv_of_volumes, skip_header=False)
    # Int labels.
    files = [(x, 0) for (x, _) in files]
    invalid = io.verify_features_labels(
        files, volume_shape=(8, 8, 8), num_parallel_calls=1
    )
    assert not invalid
    invalid = io.verify_features_labels(
        files, volume_shape=(12, 12, 8), num_parallel_calls=1
    )
    assert all(invalid)
    # Float labels.
    files = [(x, 1.0) for (x, _) in files]
    invalid = io.verify_features_labels(
        files, volume_shape=(8, 8, 8), num_parallel_calls=1
    )
    assert not invalid
    invalid = io.verify_features_labels(
        files, volume_shape=(12, 12, 8), num_parallel_calls=1
    )
    assert all(invalid)


def test_is_gzipped(tmp_path):
    filename = str(tmp_path / "test.gz")
    with gzip.GzipFile(filename, "w") as f:
        f.write("i'm more than a test!".encode())
    assert io._is_gzipped(filename)

    with open(filename, "w") as f:
        f.write("i'm just a test...")
    assert not io._is_gzipped(filename)
