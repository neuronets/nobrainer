import random

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from .utils import csv_of_volumes  # noqa: F401
from .. import io, tfrecord


def test_write_read_volume_labels(csv_of_volumes, tmp_path):  # noqa: F811
    files = io.read_csv(csv_of_volumes, skip_header=False)
    filename_template = str(tmp_path / "data-{shard:03d}.tfrecords")
    examples_per_shard = 12
    tfrecord.write(
        files,
        filename_template=filename_template,
        examples_per_shard=examples_per_shard,
        processes=1,
    )

    paths = list(tmp_path.glob("data-*.tfrecords"))
    paths = sorted(paths)
    assert len(paths) == 9
    assert (tmp_path / "data-008.tfrecords").is_file()

    dset = tf.data.TFRecordDataset(list(map(str, paths)), compression_type="GZIP")
    dset = dset.map(
        tfrecord.parse_example_fn(volume_shape=(8, 8, 8), scalar_label=False)
    )

    for ref, test in zip(files, dset):
        x, y = ref
        x, y = io.read_volume(x), io.read_volume(y)
        assert_array_equal(x, test[0])
        assert_array_equal(y, test[1])

    with pytest.raises(ValueError):
        tfrecord.write(
            files, filename_template="data/foobar-{}.tfrecords", examples_per_shard=4
        )


def test_write_read_volume_labels_all_processes(csv_of_volumes, tmp_path):  # noqa: F811
    files = io.read_csv(csv_of_volumes, skip_header=False)
    filename_template = str(tmp_path / "data-{shard:03d}.tfrecords")
    examples_per_shard = 12
    tfrecord.write(
        files,
        filename_template=filename_template,
        examples_per_shard=examples_per_shard,
        processes=None,
    )

    paths = list(tmp_path.glob("data-*.tfrecords"))
    paths = sorted(paths)
    assert len(paths) == 9
    assert (tmp_path / "data-008.tfrecords").is_file()

    dset = tf.data.TFRecordDataset(list(map(str, paths)), compression_type="GZIP")
    dset = dset.map(
        tfrecord.parse_example_fn(volume_shape=(8, 8, 8), scalar_label=False)
    )

    for ref, test in zip(files, dset):
        x, y = ref
        x, y = io.read_volume(x), io.read_volume(y)
        assert_array_equal(x, test[0])
        assert_array_equal(y, test[1])

    with pytest.raises(ValueError):
        tfrecord.write(
            files, filename_template="data/foobar-{}.tfrecords", examples_per_shard=4
        )


def test_write_read_float_labels(csv_of_volumes, tmp_path):  # noqa: F811
    files = io.read_csv(csv_of_volumes, skip_header=False)
    files = [(x, random.random()) for x, _ in files]
    filename_template = str(tmp_path / "data-{shard:03d}.tfrecords")
    examples_per_shard = 12
    tfrecord.write(
        files,
        filename_template=filename_template,
        examples_per_shard=examples_per_shard,
        processes=1,
    )

    paths = list(tmp_path.glob("data-*.tfrecords"))
    paths = sorted(paths)
    assert len(paths) == 9
    assert (tmp_path / "data-008.tfrecords").is_file()

    dset = tf.data.TFRecordDataset(list(map(str, paths)), compression_type="GZIP")
    dset = dset.map(
        tfrecord.parse_example_fn(volume_shape=(8, 8, 8), scalar_label=True)
    )

    for ref, test in zip(files, dset):
        x, y = ref
        x = io.read_volume(x)
        assert_array_equal(x, test[0])
        assert_array_equal(y, test[1])


def test_write_read_int_labels(csv_of_volumes, tmp_path):  # noqa: F811
    files = io.read_csv(csv_of_volumes, skip_header=False)
    files = [(x, random.randint(0, 9)) for x, _ in files]
    filename_template = str(tmp_path / "data-{shard:03d}.tfrecords")
    examples_per_shard = 12
    tfrecord.write(
        files,
        filename_template=filename_template,
        examples_per_shard=examples_per_shard,
        processes=1,
    )

    paths = list(tmp_path.glob("data-*.tfrecords"))
    paths = sorted(paths)
    assert len(paths) == 9
    assert (tmp_path / "data-008.tfrecords").is_file()

    dset = tf.data.TFRecordDataset(list(map(str, paths)), compression_type="GZIP")
    dset = dset.map(
        tfrecord.parse_example_fn(volume_shape=(8, 8, 8), scalar_label=True)
    )

    for ref, test in zip(files, dset):
        x, y = ref
        x = io.read_volume(x)
        assert_array_equal(x, test[0])
        assert_array_equal(y, test[1])


def test__is_int_or_float():
    assert tfrecord._is_int_or_float(10)
    assert tfrecord._is_int_or_float(10.0)
    assert tfrecord._is_int_or_float("10")
    assert tfrecord._is_int_or_float("10.00")
    assert not tfrecord._is_int_or_float("foobar")
    assert tfrecord._is_int_or_float(np.ones(1))
    assert not tfrecord._is_int_or_float(np.ones(10))


def test__dtype_to_bytes():
    np_tf_dt = [
        (np.uint8, tf.uint8, b"uint8"),
        (np.uint16, tf.uint16, b"uint16"),
        (np.uint32, tf.uint32, b"uint32"),
        (np.uint64, tf.uint64, b"uint64"),
        (np.int8, tf.int8, b"int8"),
        (np.int16, tf.int16, b"int16"),
        (np.int32, tf.int32, b"int32"),
        (np.int64, tf.int64, b"int64"),
        (np.float16, tf.float16, b"float16"),
        (np.float32, tf.float32, b"float32"),
        (np.float64, tf.float64, b"float64"),
    ]

    for npd, tfd, dt in np_tf_dt:
        npd = np.dtype(npd)
        assert tfrecord._dtype_to_bytes(npd) == dt
        assert tfrecord._dtype_to_bytes(tfd) == dt

    assert tfrecord._dtype_to_bytes("float32") == b"float32"
    assert tfrecord._dtype_to_bytes("foobar") == b"foobar"
