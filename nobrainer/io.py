"""Input/output methods."""
import csv
import functools
import math
import multiprocessing
import os

import nibabel as nib
import numpy as np
import tensorflow as tf

from nobrainer.utils import _get_all_cpus

_TFRECORDS_FEATURES_DTYPE = 'float32'


def read_csv(filepath, skip_header=True, delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]


def read_mapping(filepath, skip_header=True, delimiter=','):
    """Read CSV to dictionary, where first column becomes keys and second
    columns becomes values. Other columns are ignored. Keys and values are
    coerced to integers.
    """
    mapping = read_csv(filepath, skip_header=skip_header, delimiter=delimiter)
    if not all(map(lambda r: len(r) >= 2, mapping)):
        raise ValueError("not all rows in the mapping have at least 2 values")
    try:
        return {int(row[0]): int(row[1]) for row in mapping}
    except ValueError:
        raise ValueError(
            "mapping values must be integers but non-integer encountered")


def read_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data from a neuroimaging file."""
    img = nib.load(filepath)
    data = img.get_fdata(caching='unchanged')
    if dtype is not None:
        data = data.astype(dtype)
    return data if not return_affine else (data, img.affine)


def convert(volume_filepaths, tfrecords_template="tfrecords/data_shard-{shard:03d}.tfrecords", volumes_per_shard=100, to_ras=True, gzip_compressed=True, num_parallel_calls=None, verbose=1):
    """Convert a list of features and labels volumes to TFRecords. The volume
    data and the affine for each volume are saved, so the original images can
    be recreated. This method uses multiple cores but can still take a considerable
    amount of time for large datasets. Gzip compression should be used, as it
    dramatically decreases the size of the resulting tfrecords file.

    This method will save multiple TFRecords files if the length of `volume_filepaths` is
    greater than `volumes_per_shard`. Sharding the data into multiple TFRecords files
    is beneficial for at least two reasons:

        1. Allows for better shuffling of large datasets because we can shuffle
            the files and not only the underlying data.
        2. Enables use of parallel reading of TFRecords files with the `tf.data` API,
            which can decrease overall data processing time.

    For example, if one is converting 100 pairs of files (i.e., length of
    `volume_filepaths` is 100) and `volumes_per_shard` is 40, then three
    TFRecords files will be created. The first 40 pairs of files will be in the
    first shard, the next 40 pairs will be in the second shard, and the last
    20 pairs will be in the third shard.

    Parameters
    ----------
    volume_filepaths: nested list. Every sublist in the list should contain two
        items: path to features volume and path to labels volume, in that order.
    tfrecords_template: string template, path to save new tfrecords file with the string formatting
        key 'shard'. The shard index is entered
        Extension should be '.tfrecords'.
    volumes_per_shard: int, number of pairs of volumes per tfrecords file.
    to_ras: boolean, if true, reorient volumes to RAS with `nibabel.as_closest_canonical`.
    gzip_compressed: boolean, if true, compress data with Gzip. This is highly
        recommended, as it dramatically reduces file size.
    num_parallel_calls: int, number of processes to use for multiprocessing. If
        None, will use all available processes.
    verbose: {0, 1, 2}, verbosity of the progress bar. 0 is silent, 1 is verbose,
        and 2 is semi-verbose.

    Returns
    -------
    None
    """
    try:
        tfrecords_template.format(shard=3)
    except Exception:
        raise ValueError("invalid 'tfrecords_template'. This template must contain the key 'shard'.")

    tfrecords_template = os.path.abspath(tfrecords_template)
    _dirname = os.path.dirname(tfrecords_template)
    if not os.path.exists(_dirname):
        raise ValueError("directory does not exist: {}".format(_dirname))

    n_shards = math.ceil(len(volume_filepaths) / volumes_per_shard)
    # Include the unique tfrecords filepath for each file, because that's what
    # the map function expects.
    volume_filepaths_shards = [
        [tfrecords_template.format(shard=idx), shard.tolist()]
        for idx, shard in enumerate(np.array_split(volume_filepaths, n_shards))]
    map_fn = functools.partial(
        _convert, to_ras=to_ras, gzip_compressed=gzip_compressed)

    print("Converting {} pairs of files to {} TFRecords.".format(len(volume_filepaths), len(volume_filepaths_shards)))
    progbar = tf.keras.utils.Progbar(len(volume_filepaths_shards), verbose=verbose)
    progbar.update(0)
    if num_parallel_calls is None:
        num_parallel_calls = _get_all_cpus()

    if num_parallel_calls == 1:
        for vf in volume_filepaths_shards:
            map_fn(vf)
            progbar.add(1)
    else:
        with multiprocessing.Pool(num_parallel_calls) as p:
            for _ in p.imap(map_fn, volume_filepaths_shards, chunksize=2):
                progbar.add(1)


def _convert(tfrecords_path_volume_filepaths, to_ras=True, gzip_compressed=True):
    """Convert a nested list of files to one TFRecords file. This function is
    not intended for users. It is used as part of the multiprocessing function `convert`.

    Parameters
    ----------
    tfrecords_path_volume_filepaths: list of length 2. The first item should be the path to the TFRecords file to create,
        and the second item should be a nested list of medical imaging filepaths to convert. Each item in this nested
        list should have two items, where the first is the path to features and the second is the path to labels.
        This argument is structured this way to be compatible with multiprocessing map and imap methods.
    to_ras: boolean, if true, reorient the volumes to RAS before saving to TFRecords.
    gzip_compressed: boolean, if true, save to a Gzip-compressed file. It is highly recommended to use compression,
        as it dramatically reduces the size of the resulting TFRecords file.

    Returns
    -------
    None
    """
    tfrecords_path, volume_filepaths = tfrecords_path_volume_filepaths

    def _make_one_example(features_filepath, labels_filepath):
        """Create a `tf.train.Example` instance of the given arrays of volumetric features and labels."""
        dtype = _TFRECORDS_FEATURES_DTYPE
        x = nib.load(features_filepath)
        if to_ras:
            x = nib.as_closest_canonical(x)
        xdata = x.get_fdata(caching='unchanged', dtype=dtype)
        y = nib.load(labels_filepath)
        if to_ras:
            y = nib.as_closest_canonical(y)
        ydata = y.get_fdata(caching='unchanged', dtype=dtype)

        def bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        feature = {
            'volume': bytes_feature(xdata.ravel().tostring()),
            'volume_affine': bytes_feature(x.affine.astype(dtype).ravel().tostring()),
            'label': bytes_feature(ydata.ravel().tostring()),
            'label_affine': bytes_feature(y.affine.astype(dtype).ravel().tostring()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    if gzip_compressed:
        options = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)
    else:
        options = None

    with tf.io.TFRecordWriter(tfrecords_path, options=options) as writer:
        for fpath, lpath in volume_filepaths:
            this_example = _make_one_example(features_filepath=fpath, labels_filepath=lpath)
            writer.write(this_example.SerializeToString())


def get_parse_fn(volume_shape=(256, 256, 256), include_affines=False):
    """Return a callable that can parse examples from a TFRecords file. It is
    assumed that each eaxmple in the TFRecords file has the keys 'volume' and
    'label'. If 'include_affines' is true, each example must also have the keys
    'volume_affine' and 'label_affine'.

    Parameters
    ----------
    volume_shape: tuple of 3 ints, the shape of each volume in the TFRecords file.
    include_affines: boolean, if true, return affine along with volumetric data.

    Returns
    -------
    Callable used to parse a TFRecords file. This callable takes one serialized
    example as inputs. If `include_affines` is false, this callable returns two
    tensors: the tensor of the features volume and the tensor of the labels
    volume. If `include_affines` if true, this callable returns four tensors:
    the features tensor, the labels tensor, the features affine tensor, and the
    labels affine tensor.
    """
    affine_shape = (4, 4)

    def parse_record(serialized):
        """Return tensors from serialized TFRecords example."""
        features = {
            # Would using a pre-defined shape (dependent on volume shape)
            # improve read performance?
            'volume': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'label': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        }

        if include_affines:
            features['volume_affine'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
            features['label_affine'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

        def decode(s):
            return tf.io.decode_raw(s, _TFRECORDS_FEATURES_DTYPE)

        example = tf.io.parse_single_example(serialized=serialized, features=features)
        volume = tf.reshape(decode(example['volume']), shape=volume_shape)
        label = tf.reshape(decode(example['label']), shape=volume_shape)

        if include_affines:
            volume_affine = tf.reshape(decode(example['volume_affine']), shape=affine_shape)
            label_affine = tf.reshape(decode(example['label_affine']), shape=affine_shape)
            return volume, label, volume_affine, label_affine,
        else:
            return volume, label

    return parse_record


def verify_features_labels(volume_filepaths, volume_shape=(256, 256, 256), check_shape=True, check_labels_int=True, check_labels_gte_zero=True, num_parallel_calls=None, verbose=1):
    """Verify a list of files. This function is meant to be run before
    converting volumes to TFRecords.

    Parameters
    ----------
    volume_filepaths: nested list. Every sublist in the list should contain two
        items: path to features volume and path to labels volume, in that order.
    volume_shape: tuple of three ints. Shape that both volumes should be.
    check_shape: boolean, if true, validate that the shape of both volumes is
        equal to 'volume_shape'.
    check_labels_int: boolean, if true, validate that every labels volume is an
        integer type or can be safely converted to an integer type.
    check_labels_gte_zero: boolean, if true, validate that every labels volume
        has values greater than or equal to zero.
    num_parallel_calls: int, number of processes to use for multiprocessing. If
        None, will use all available processes.
    verbose: {0, 1, 2}, verbosity of the progress bar. 0 is silent, 1 is verbose,
        and 2 is semi-verbose.

    Returns
    -------
    List of invalid pairs of filepaths. If the list is empty, all filepaths are
    valid.
    """

    for pair in volume_filepaths:
        if len(pair) != 2:
            raise ValueError(
                "all items in 'volume_filepaths' must have length of 2, but"
                " found at least one item with lenght != 2.")
        if not os.path.exists(pair[0]):
            raise ValueError("file does not exist: {}".format(pair[0]))
        if not os.path.exists(pair[1]):
            raise ValueError("file does not exist: {}".format(pair[1]))

    print("Verifying {} pairs of volumes".format(len(volume_filepaths)))
    progbar = tf.keras.utils.Progbar(len(volume_filepaths), verbose=verbose)
    progbar.update(0)
    map_fn = functools.partial(_verify_features_labels_pair, volume_shape=volume_shape, check_shape=check_shape, check_labels_int=check_labels_int, check_labels_gte_zero=check_labels_gte_zero)
    if num_parallel_calls is None:
        num_parallel_calls = _get_all_cpus()

    outputs = []
    if num_parallel_calls == 1:
        for vf in volume_filepaths:
            valid = map_fn(vf)
            outputs.append(valid)
            progbar.add(1)
    else:
        with multiprocessing.Pool(num_parallel_calls) as p:
            for valid in p.imap(map_fn, volume_filepaths, chunksize=2):
                outputs.append(valid)
                progbar.add(1)
    invalid_files = [pair for valid, pair in zip(outputs, volume_filepaths) if not valid]
    return invalid_files


def _verify_features_labels_pair(pair_of_paths, *, volume_shape, check_shape, check_labels_int, check_labels_gte_zero):
    """Verify a pair of features and labels volumes."""
    x = nib.load(pair_of_paths[0])
    y = nib.load(pair_of_paths[1])
    if check_shape:
        if not volume_shape:
            raise ValueError("`volume_shape` must be specified if `check_shape` is true.")
        if x.shape != volume_shape:
            return False
        if x.shape != y.shape:
            return False
    if check_labels_int:
        # Quick check of integer type.
        if not np.issubdtype(y.dataobj.dtype, np.integer):
            return False
        y = y.get_fdata(caching='unchanged', dtype=np.float32)
        # Longer check that all values in labels can be cast to int.
        if not np.all(np.mod(y, 1) == 0):
            return False
    if check_labels_gte_zero:
        if not np.all(y >= 0):
            return False
    return True


def _is_gzipped(filepath):
    """Return True if the file is gzip-compressed, False otherwise."""
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'
