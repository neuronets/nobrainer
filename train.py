#!/usr/bin/env python3
"""Script to train highres3dnet model.

The input CSV must have two columns:
    1. filepaths of features
    2. filepaths of corresponding labels

TODO
----
- Make this script more general. Ideally, one could drop in their model and
    loss function.
- Move some common methods (eg, i/o) to dedicated modules.
- Dice coefficient for class 1 (brainmask) is sometimes NaN.
- Input of 1 * 128**3 is too large for 1080ti. This seems to be related to the
    `input_fn` used.
- Remove pandas as a dependency. Make pure python reader that accepts CSV or
    TSV as input.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

import nobrainer

params = {
    'csv_filepath': '',
    'num_classes': 2,
    'learning_rate': 0.001,
    'data_shape': (256, 256, 256),
    'block_shape': (64, 64, 64),
    'batch_size': 4,
    'repeat': 0,  # number of times to repeat (0 for one epoch)
}

df = pd.read_csv(params['csv_filepath'])

# Data types of labels (x) and features (y).
DT_X = 'float32'
DT_Y = 'int32'
_DT_X_NP = np.dtype(DT_X)
_DT_X_TF = tf.as_dtype(DT_X)
_DT_Y_NP = np.dtype(DT_Y)
_DT_Y_TF = tf.as_dtype(DT_Y)


def model_fn(features, labels, mode, params):

    logits = nobrainer.models.highres3dnet(
        features=features,
        num_classes=params['num_classes'],
        mode=mode,
        save_to_histogram=False,
    )

    predictions = tf.argmax(logits, axis=-1)
    # probabilities = tf.nn.softmax(logits, -1)  # unused at the moment.

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
        )

    # loss function
    # QUESTION: which loss function should we use?
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits,
    )
    # cross_entropy.shape == (batch_size, *block_shape)
    loss = tf.reduce_mean(cross_entropy)

    # training op
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate'],
    )
    global_step = tf.train.get_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    _inp = [
        tf.one_hot(labels, depth=params['num_classes'], axis=-1),
        tf.one_hot(predictions, depth=params['num_classes'], axis=-1),
        tf.constant(params['num_classes'], dtype=tf.uint16)
    ]
    dice_coefficients = tf.py_func(
        func=nobrainer.metrics.dice_coefficient_by_class_numpy,
        inp=_inp,
        Tout=tf.float32,
        stateful=False,
        name='dice',
    )

    # Record Dice coefficient for each class.
    # Idea from DLTK.
    #
    # TODO: sometimes NaN is returned for Dice of class 1 (brainmask). This
    # issue comes and goes. Why?
    # Relevant: `RuntimeWarning: invalid value encountered in double_scalars`
    for ii in range(params['num_classes']):
        tf.summary.scalar('dice_class_{}'.format(ii), dice_coefficients[ii])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
    )


def load_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data."""
    img = nib.load(filepath)
    data = np.asarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    img.uncache()
    return data if not return_affine else (data, img.affine)


def as_blocks(a, block_shape):
    """Return new array of shape `(n_blocks, *block_shape)`."""
    orig_shape = np.asarray(a.shape)
    blocks = orig_shape // block_shape
    inter_shape = tuple(e for tup in zip(blocks, block_shape) for e in tup)
    new_shape = (-1,) + block_shape
    perm = (0, 2, 4, 1, 3, 5)  # TODO: generalize this.
    return a.reshape(inter_shape).transpose(perm).reshape(new_shape)


def _load_feature_label():
    """Yield tuples of numpy arrays `(features, labels)`."""
    for idx, (features_fp, labels_fp) in df.iterrows():

        tf.logging.info("Reading pair number {}".format(idx))

        features = load_volume(features_fp, dtype=_DT_X_NP)
        features = as_blocks(features, params['block_shape'])
        features = np.expand_dims(features, -1)

        labels = load_volume(labels_fp, dtype=_DT_Y_NP)
        labels = as_blocks(labels, params['block_shape'])

        n_blocks = features.shape[0]
        batch_size = params['batch_size']

        if batch_size > n_blocks:
            raise ValueError(
                "Batch size must be less than or equal to the number of"
                " blocks. Got batch size `{}` and {} blocks"
                .format(batch_size, n_blocks)
            )

        iter_range = n_blocks / batch_size
        if not iter_range.is_integer():
            raise ValueError(
                "Batch size must be a factor of number of blocks. Got batch"
                " size `{}` and {} blocks".format(batch_size, n_blocks)
            )

        # Yield non-overlapping batches of blocks.
        for ii in range(int(iter_range)):
            tf.logging.debug("Yielding batch {}".format(ii))
            _start = int(ii * batch_size)
            _end = _start + batch_size
            _slice = slice(_start, _end)
            yield (
                features[_slice, Ellipsis],
                labels[_slice, Ellipsis],
            )


def input_fn():
    """Input function meant to be used with `tf.estimator.Estimator`."""
    block_shape = params['block_shape']
    output_types = (_DT_X_TF, _DT_Y_TF)
    output_shapes = (
        tf.TensorShape((None, *block_shape, 1)),
        tf.TensorShape((None, *block_shape))
    )

    dset = tf.data.Dataset.from_generator(
        generator=_load_feature_label,
        output_types=output_types,
        output_shapes=output_shapes,
    )

    epochs = params.get('epochs', None)
    dset = dset.repeat(epochs)

    return dset


def main():
    """Train estimator."""
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True

    with tf.Session(config=_config) as sess:

        _runconfig = tf.estimator.RunConfig(
            save_summary_steps=50, keep_checkpoint_max=20,
        )

        model = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir='./logs',
            config=_runconfig,
            params=params,
        )

        model.train(input_fn=input_fn)


if __name__ == '__main__':
    main()
