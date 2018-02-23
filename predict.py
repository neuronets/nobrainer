#!/usr/bin/env python3

import nibabel as nib
import numpy as np
import tensorflow as tf

import nobrainer


def from_blocks(a, output_shape):
    """Initial implementation of function to go from blocks to full volume."""
    return (
        a.reshape((2, 2, 2, 128, 128, 128))
        .transpose((0, 3, 1, 4, 2, 5))
        .reshape(output_shape))


def predict(filepath):
    """Return Nibabel Nifti image of predicted brainmask for neuroimaging file
    `filepath`.
    """
    x, x_affine = nobrainer.io.load_volume(
        filepath, dtype='float32', return_affine=True
    )

    assert x.shape == (256, 256, 256), "shape must be (256, 256, 256)."

    x = nobrainer.preprocessing.as_blocks(x, (128, 128, 128))
    x = np.expand_dims(x, -1)
    x = nobrainer.preprocessing.normalize_zero_one(x)

    inputs = tf.estimator.inputs.numpy_input_fn(
        x=x, shuffle=False, batch_size=4
    )

    model = nobrainer.models.MeshNet(
        2, model_dir='models-hdf5/meshnet_20180219-220005'
    )

    generator = model.predict(input_fn=inputs)
    predictions = np.zeros((8, 128, 128, 128))

    for ii, block in enumerate(generator):
        predictions[ii, Ellipsis] = block

    predictions = from_blocks(predictions, (256, 256, 256))

    return nib.Nifti1Image(predictions, x_affine)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        raise ValueError("input and output filepaths must be specified.")

    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    brainmask_nifti = predict(input_filepath)

    brainmask_nifti.to_filename(output_filepath)

