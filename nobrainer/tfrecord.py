import numpy as np
import tensorflow as tf

from nobrainer.io import read_volume

_TFRECORDS_DTYPE = "float32"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _dtype_to_bytes(dtype):
    if isinstance(dtype, str):
        return dtype.encode('utf-8')
    try:
        s = str(dtype.name)
    except AttributeError:
        s = str(dtype)
    return s.encode('utf-8')


def _get_feature_dict(x, affine):
    x = np.asarray(x)
    x = x.astype(_TFRECORDS_DTYPE)

    feature = {
        "feature/value": _bytes_feature(x.tobytes()),
        "feature/dtype": _bytes_feature(_dtype_to_bytes(x.dtype)),
        "feature/rank": _int64_feature(x.ndim)}

    for j, s in enumerate(x.shape):
        feature["feature/shape/dim{}".format(j)] = _int64_feature(s)

    if x.ndim:
        feature["feature/shape"] = _bytes_feature(np.array(x.shape).astype(_TFRECORDS_DTYPE).tobytes())

    affine = np.asarray(affine)
    affine = affine.astype(_TFRECORDS_DTYPE)
    feature.update({
        "feature/affine/value": _bytes_feature(affine.tobytes()),
        "feature/affine/dtype": _bytes_feature(_dtype_to_bytes(affine.dtype)),
        "feature/affine/rank": _int64_feature(affine.ndim),
    })
    for j, s in enumerate(affine.shape):
        feature["feature/affine/shape/dim{}".format(j)] = _int64_feature(s)

    return feature


def _get_label_dict(y, affine=None):
    y = np.asarray(y)
    y = y.astype(_TFRECORDS_DTYPE)

    label = {
        "label/value": _bytes_feature(y.tobytes()),
        "label/dtype": _bytes_feature(_dtype_to_bytes(y.dtype)),
        "label/rank": _int64_feature(y.ndim)}

    for j, s in enumerate(y.shape):
        label["label/shape/dim{}".format(j)] = _int64_feature(s)
    if y.ndim:
        feature["feature/shape"] = _bytes_feature(np.array(y.shape).astype(_TFRECORDS_DTYPE).tobytes())

    if affine is None and y.ndim != 0:
        raise ValueError("Affine is required when label is not scalar.")

    if affine is not None:
        affine = np.asarray(affine)
        affine = affine.astype(_TFRECORDS_DTYPE)
        label.update({
            "label/affine/value": _bytes_feature(affine.tobytes()),
            "label/affine/dtype": _bytes_feature(_dtype_to_bytes(affine.dtype)),
            "label/affine/rank": _int64_feature(affine.ndim),
        })
        for j, s in enumerate(affine.shape):
            label["label/affine/shape/dim{}".format(j)] = _int64_feature(s)

    return label


def _to_proto(feature, label, feature_affine, label_affine=None):
    """Return instance of `tf.train.Example` of feature and label."""
    d = _get_feature_dict(x=feature, affine=feature_affine)
    d.update(_get_label_dict(y=label, affine=label_affine))
    return tf.train.Example(features=tf.train.Features(feature=d))


def _iter_proto_strings(features_labels, scalar_label=False):
    """

    """
    if scalar_label:
        for xpath, y in features_labels:
            x, affine = read_volume(xpath, return_affine=True, dtype=_TFRECORDS_DTYPE)
            proto = _to_proto(feature=x, label=y, feature_affine=affine, label_affine=None)
            yield proto.SerializeToString()
    else:
        for xpath, ypath in features_labels:
            x, affine_x = nobrainer.io.read_volume(xpath, return_affine=True, dtype=_TFRECORDS_DTYPE)
            y, affine_y = nobrainer.io.read_volume(ypath, return_affine=True, dtype=_TFRECORDS_DTYPE)
            proto = _to_proto(feature=x, label=y, feature_affine=affine_x, label_affine=affine_y)
            yield proto.SerializeToString()


def _write_tfrecords(protobuf_iterator, filename, compressed=True):
    """
    protobuf_iterator: iterator, iterator which yields protocol-buffer serialized strings.
    """
    if compressed:
        options = tf.io.TFRecordOptions(compression_type="GZIP")
    else:
        options = None
    with tf.io.TFRecordWriter('foobar.tfrec', options=options) as f:
        for proto_string in protobuf_iterator:
            f.write(proto_string)


@tf.function
def _parse_example(serialized):
    features = {
        "feature/shape": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "feature/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label/value": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label/rank": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    e = tf.io.parse_single_example(serialized=serialized, features=features)

    x = tf.io.decode_raw(e["feature/value"], _TFRECORDS_DTYPE)
    y = tf.io.decode_raw(e["label/value"], _TFRECORDS_DTYPE)
    xshape = tf.cast(tf.io.decode_raw(e["feature/shape"], _TFRECORDS_DTYPE), tf.int32)
    x = tf.reshape(x, shape=xshape)

    if e["label/rank"] != 0:
        ee = tf.io.parse_single_example(serialized=serialized, features={
            "label/shape": tf.io.FixedLenFeature(shape=[], dtype=tf.string)})
        yshape = tf.cast(tf.io.decode_raw(ee["label/shape"], _TFRECORDS_DTYPE), tf.int32)
        y = tf.reshape(y, shape=yshape)

    return x, y
