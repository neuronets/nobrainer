import functools
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import skimage.transform
import tensorflow as tf

from .io import read_volume
from .utils import get_num_parallel

_TFRECORDS_DTYPE = "float32"


def write(
    features_labels,
    filename_template,
    examples_per_shard,
    to_ras=True,
    compressed=True,
    processes=None,
    chunksize=1,
    multi_resolution=False,
    resolutions=None,
    verbose=1,
):
    """Write features and labels to TFRecord files.

    This method supports the use of scalar or n-D labels. All labels, however, must be
    scalar, or all labels must be n-D. If labels are n-D, their shape must be the same
    as the shape of features.

    Parameters
    ----------
    features_labels: nested sequence, pairs of features and labels. If the labels are
        n-D, each pair should be `(path_to_feature, path_to_label)`. If labels are
        scalar, each pair should be `(path_to_feature, scalar_label)`.
    filename_template: str, the path to which TFRecord files should be written. A string
        formatting key `shard` should be included to indicate the unique TFRecord file
        when writing to multiple TFRecord files. For example,
        `data_shard-{shard:03d}.tfrec`.
    examples_per_shard: int, number of pairs of `(feature, label)` per TFRecord file.
    to_ras: boolean, reorient volumes to RAS before writing to TFRecord file.
    compressed: boolean, if `True`, gzip compress TFRecord file. Highly recommended.
    processes: int, number of processes for multiprocessing. This is useful when
        writing to multiple TFRecord files (i.e.,
        `examples_per_shard` < `len(features_labels)`). If `None`, uses all available
        cores.
    chunksize: int, multiprocessing chunksize.
    multi_resolution: boolean, if `True`, different tfrecords for each resolution in each shard
    resolutions: list of ints, if multi_resolution is `True`, set resolutions for
        which tfrecords are created. For example, [4, 8, 16, 32, 64, 128, 256]
    verbose: int, if 1, print progress bar. If 0, print nothing.
    """
    n_examples = len(features_labels)
    n_shards = math.ceil(n_examples / examples_per_shard)
    shards = np.array_split(features_labels, n_shards)

    # Test that the `filename_template` has a `shard` formatting key.
    try:
        filename_template.format(shard=0)
    except Exception:
        raise ValueError(
            "`filename_template` must include a string formatting key 'shard' that"
            " accepts an integer."
        )
    # Test if resolutions is set
    if multi_resolution and not isinstance(resolutions, list):
        raise ValueError(
            "`resolutions` must be set with a list of ints indicating resolutions for tfrecords"
        )

    # This is the object that returns a protocol buffer string of the feature and label
    # on each iteration. It is pickle-able, unlike a generator.
    proto_iterators = [
        _ProtoIterator(s, multi_resolution=multi_resolution, resolutions=resolutions)
        for s in shards
    ]
    # Set up positional arguments for the core writer function.
    iterable = [
        (p, filename_template.format(shard=j)) for j, p in enumerate(proto_iterators)
    ]
    # Set keyword arguments so the resulting function accepts one positional argument.
    map_fn = functools.partial(
        _write_tfrecords,
        compressed=True,
        multi_resolution=multi_resolution,
        resolutions=resolutions,
    )

    # This is a hack to allow multiprocessing to pickle
    # the __writer_func object. Pickles don't like local functions.
    global __writer_func

    def __writer_func(iterator_filename):
        iterator, filename = iterator_filename
        map_fn(protobuf_iterator=iterator, filename=filename)

    progbar = tf.keras.utils.Progbar(target=len(iterable), verbose=verbose)
    progbar.update(0)
    if processes is None:
        processes = get_num_parallel()
    with mp.get_context("fork").Pool(processes) as p:
        for _ in p.imap_unordered(
            __writer_func, iterable=iterable, chunksize=chunksize
        ):
            progbar.add(1)


def parse_example_fn(volume_shape, scalar_label=False):
    """Return function that can be used to read TFRecord file into tensors.

    Parameters
    ----------
    volume_shape: sequence, the shape of the feature data. If `scalar_label` is `False`,
        this also corresponds to the shape of the label data.
    scalar_label: boolean, if `True`, label is a scalar. If `False`, label must be the
        same shape as feature data.

    Returns
    -------
    Function with which a TFRecord file can be parsed.
    """

    @tf.function
    def parse_example(serialized):
        """Parse one example from a TFRecord file made with Nobrainer.

        Parameters
        ----------
        serialized: str, serialized proto message.

        Returns
        -------
        Tuple of two tensors. If `scalar_label` is `False`, both tensors have shape
        `volume_shape`. Otherwise, the first tensor has shape `volume_shape`, and the
        second is a scalar tensor.
        """
        features = {
            "feature/shape": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "feature/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "label/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "label/rank": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        }
        e = tf.io.parse_single_example(serialized=serialized, features=features)
        x = tf.io.decode_raw(e["feature/value"], _TFRECORDS_DTYPE)
        y = tf.io.decode_raw(e["label/value"], _TFRECORDS_DTYPE)
        # TODO: this line does not work. The shape cannot be determined
        # dynamically... for now.
        # xshape = tf.cast(
        #     tf.io.decode_raw(e["feature/shape"], _TFRECORDS_DTYPE), tf.int32)
        x = tf.reshape(x, shape=volume_shape)
        if not scalar_label:
            y = tf.reshape(y, shape=volume_shape)
        else:
            y = tf.reshape(y, shape=[1])
        return x, y

    return parse_example


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _dtype_to_bytes(dtype):
    if isinstance(dtype, str):
        return dtype.encode("utf-8")
    try:
        s = str(dtype.name)
    except AttributeError:
        s = str(dtype)
    return s.encode("utf-8")


def _get_feature_dict(x, affine):
    x = np.asarray(x)
    x = x.astype(_TFRECORDS_DTYPE)

    feature = {
        "feature/value": _bytes_feature(x.tobytes()),
        "feature/dtype": _bytes_feature(_dtype_to_bytes(x.dtype)),
        "feature/rank": _int64_feature(x.ndim),
    }

    for j, s in enumerate(x.shape):
        feature["feature/shape/dim{}".format(j)] = _int64_feature(s)

    if x.ndim:
        feature["feature/shape"] = _bytes_feature(
            np.array(x.shape).astype(_TFRECORDS_DTYPE).tobytes()
        )

    affine = np.asarray(affine)
    affine = affine.astype(_TFRECORDS_DTYPE)
    feature.update(
        {
            "feature/affine/value": _bytes_feature(affine.tobytes()),
            "feature/affine/dtype": _bytes_feature(_dtype_to_bytes(affine.dtype)),
            "feature/affine/rank": _int64_feature(affine.ndim),
        }
    )
    for j, s in enumerate(affine.shape):
        feature["feature/affine/shape/dim{}".format(j)] = _int64_feature(s)

    return feature


def _get_label_dict(y, affine=None):
    y = np.asarray(y)
    y = y.astype(_TFRECORDS_DTYPE)

    label = {
        "label/value": _bytes_feature(y.tobytes()),
        "label/dtype": _bytes_feature(_dtype_to_bytes(y.dtype)),
        "label/rank": _int64_feature(y.ndim),
    }

    for j, s in enumerate(y.shape):
        label["label/shape/dim{}".format(j)] = _int64_feature(s)
    if y.ndim:
        label["feature/shape"] = _bytes_feature(
            np.array(y.shape).astype(_TFRECORDS_DTYPE).tobytes()
        )

    if affine is None and y.ndim != 0:
        raise ValueError("Affine is required when label is not scalar.")

    if affine is not None:
        affine = np.asarray(affine)
        affine = affine.astype(_TFRECORDS_DTYPE)
        label.update(
            {
                "label/affine/value": _bytes_feature(affine.tobytes()),
                "label/affine/dtype": _bytes_feature(_dtype_to_bytes(affine.dtype)),
                "label/affine/rank": _int64_feature(affine.ndim),
            }
        )
        for j, s in enumerate(affine.shape):
            label["label/affine/shape/dim{}".format(j)] = _int64_feature(s)

    return label


def _to_proto(feature, label, feature_affine, label_affine=None):
    """Return instance of `tf.train.Example` of feature and label."""
    d = _get_feature_dict(x=feature, affine=feature_affine)
    d.update(_get_label_dict(y=label, affine=label_affine))
    return tf.train.Example(features=tf.train.Features(feature=d))


class _ProtoIterator:
    """Iterator of protobuf strings.

    Parameters
    ----------

    This custom iterator is used instead of a generator for use in
    multiprocessing. Generators cannot be pickled, and so cannot be used in
    multiprocessing workflows. Please see
    https://stackoverflow.com/a/7180424/5666087 for more information.
    """

    def __init__(
        self, features_labels, to_ras=True, multi_resolution=False, resolutions=None
    ):
        self.features_labels = features_labels
        self.to_ras = to_ras
        self.multi_resolution = multi_resolution
        self.resolutions = resolutions

        # Try to "intelligently" deduce if the labels are scalars or not.
        # An alternative here would be to check if these point to existing
        # files, though it is possible to have existing filenames that
        # are integers or floats.
        labels = [y for _, y in features_labels]
        self.scalar_label = _labels_all_scalar(labels)
        self._j = 0

        no_exist = []
        for x, _ in self.features_labels:
            if not Path(x).exists():
                no_exist.append(x)
        if no_exist:
            raise ValueError("Some files do not exist: {}".format(", ".join(no_exist)))

        if not self.scalar_label:
            no_exist = []
            for _, y in self.features_labels:
                if not Path(y).exists():
                    no_exist.append(y)
            if no_exist:
                raise ValueError(
                    "Some files do not exist: {}".format(", ".join(no_exist))
                )

    def __iter__(self):
        self._j = 0
        return self

    def __next__(self):
        index = self._j
        serialized = self._serialize(index)
        self._j += 1
        return serialized

    def _serialize(self, index):
        try:
            x, y = self.features_labels[index]
        except IndexError:
            raise StopIteration
        x, affine_x = read_volume(
            x, return_affine=True, dtype=_TFRECORDS_DTYPE, to_ras=self.to_ras
        )
        if self.multi_resolution:
            # only scalar label
            if not self.scalar_label:
                y = 0
            proto_dict = {}
            for resolution in self.resolutions[::-1]:
                x_res = skimage.transform.resize(
                    x,
                    output_shape=(resolution, resolution, resolution),
                    order=1,  # linear
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=True,
                )
                # TODO: Affine should be adjusted when resizing
                proto = _to_proto(
                    feature=x_res, label=y, feature_affine=affine_x, label_affine=None
                )
                proto_dict[resolution] = proto.SerializeToString()

            return proto_dict
        else:
            label_affine = None
            if not self.scalar_label:
                y, label_affine = read_volume(
                    y, return_affine=True, dtype=_TFRECORDS_DTYPE, to_ras=self.to_ras
                )
            proto = _to_proto(
                feature=x,
                label=y,
                feature_affine=affine_x,
                label_affine=label_affine,
            )
            return proto.SerializeToString()


def _write_tfrecords(
    protobuf_iterator,
    filename,
    compressed=True,
    multi_resolution=False,
    resolutions=None,
):
    """
    protobuf_iterator: iterator, iterator which yields protocol-buffer serialized
        strings.
    """
    if compressed:
        options = tf.io.TFRecordOptions(compression_type="GZIP")
    else:
        options = None
    if not multi_resolution:
        with tf.io.TFRecordWriter(filename, options=options) as f:
            for proto_string in protobuf_iterator:
                f.write(proto_string)
    else:
        tf_record_writers = {}
        filenames = [
            "{0}-res-{2:03d}{1}".format(*os.path.splitext(filename) + (res,))
            for res in resolutions
        ]
        for resolution, filename in zip(resolutions, filenames):
            tf_record_writers[resolution] = tf.io.TFRecordWriter(
                filename, options=options
            )
        for proto_string_dict in protobuf_iterator:
            for resolution in resolutions:
                tf_record_writers[resolution].write(proto_string_dict[resolution])
        [tf_record_writers[writer].close() for writer in tf_record_writers]


def _is_int_or_float(value):
    if isinstance(value, (int, float)):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        pass
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        pass
    return False


def _labels_all_scalar(labels):
    scalars = list(map(_is_int_or_float, labels))
    if any(scalars) and not all(scalars):
        raise ValueError(
            "Some labels were detected as scalars, while others were not."
            " Labels must be all scalars or all filenames of volumes."
        )
    return all(scalars)
