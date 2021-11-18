"""Tests for `nobrainer.prediction`."""

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import tensorflow as tf

from .. import prediction
from ..models.bayesian import variational_meshnet
from ..models.meshnet import meshnet


def test_predict(tmp_path):

    x = np.ones((4, 4, 4))
    img = nib.Nifti1Image(x, affine=np.eye(4))
    path = str(tmp_path / "features.nii.gz")
    img.to_filename(path)

    x2 = x * -50
    img2 = nib.Nifti1Image(x2, affine=np.eye(4))
    path2 = str(tmp_path / "features2.nii.gz")
    img2.to_filename(path2)

    model = meshnet(1, (*x.shape, 1), receptive_field=37)

    # From array.
    y_ = prediction.predict_from_array(x, model=model, block_shape=None)
    y_blocks = prediction.predict_from_array(x, model=model, block_shape=x.shape)
    y_other = prediction.predict(x, model=model, block_shape=None)
    assert isinstance(y_, np.ndarray)
    assert_array_equal(y_.shape, *x.shape)
    assert_array_equal(y_, y_other)
    assert_array_equal(y_, y_blocks)

    # From image.
    y_img = prediction.predict_from_img(img, model=model, block_shape=None)
    y_img_other = prediction.predict(img, model=model, block_shape=None)
    assert isinstance(y_img, nib.spatialimages.SpatialImage)
    assert_array_equal(y_img.shape, x.shape)
    assert_array_equal(y_img.get_fdata(caching="unchanged"), y_)
    assert_array_equal(
        y_img.get_fdata(caching="unchanged"), y_img_other.get_fdata(caching="unchanged")
    )

    # From filepath
    y_img2 = prediction.predict_from_filepath(path, model=model, block_shape=None)
    y_img2_other = prediction.predict(path, model=model, block_shape=None)
    assert isinstance(y_img, nib.spatialimages.SpatialImage)
    assert_array_equal(y_img.shape, x.shape)
    assert_array_equal(y_img.get_fdata(caching="unchanged"), y_)
    assert_array_equal(
        y_img2.get_fdata(caching="unchanged"),
        y_img2_other.get_fdata(caching="unchanged"),
    )

    # From filepaths
    gen = prediction.predict_from_filepaths(
        [path, path2], model=model, block_shape=None
    )
    y_img3 = next(gen)
    y_img4 = next(gen)
    gen_other = prediction.predict([path, path2], model=model, block_shape=None)
    y_img3_other = next(gen_other)
    y_img4_other = next(gen_other)

    assert_array_equal(
        y_img2.get_fdata(caching="unchanged"), y_img3.get_fdata(caching="unchanged")
    )
    assert_array_equal(
        y_img3.get_fdata(caching="unchanged"),
        y_img3_other.get_fdata(caching="unchanged"),
    )
    assert_array_equal(
        y_img4.get_fdata(caching="unchanged"),
        y_img4_other.get_fdata(caching="unchanged"),
    )
    assert_array_equal(y_img3.shape, x.shape)


def test_variational_predict(tmp_path):
    x = np.ones((4, 4, 4))
    img = nib.Nifti1Image(x, affine=np.eye(4))
    path = str(tmp_path / "features.nii.gz")
    img.to_filename(path)

    x2 = x * -50
    img2 = nib.Nifti1Image(x2, affine=np.eye(4))
    path2 = str(tmp_path / "features2.nii.gz")
    img2.to_filename(path2)

    model = variational_meshnet(1, (*x.shape, 1), receptive_field=37)

    # From array.
    mean, var, entropy = prediction.predict_from_array(
        x,
        model=model,
        block_shape=None,
        n_samples=2,
        return_variance=True,
        return_entropy=True,
    )
    # y_blocks = prediction.predict_from_array(x, model=model, block_shape=x.shape)
    # y_other = prediction.predict(x, model=model, block_shape=None)
    assert isinstance(mean, np.ndarray)
    assert isinstance(var, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert_array_equal(mean.shape, *x.shape)
    assert_array_equal(var.shape, *x.shape)
    assert_array_equal(entropy.shape, *x.shape)
    # assert_array_equal(y_, y_other)
    # assert_array_equal(y_, y_blocks)


def test_get_model(tmp_path):
    model = meshnet(3, (10, 10, 10, 1), receptive_field=37)
    path = str(tmp_path / "model.h5")
    model.save(path)
    assert isinstance(prediction._get_model(path), tf.keras.Model)
    assert model is prediction._get_model(model)
    with pytest.raises(ValueError):
        prediction._get_model("not a model")


@pytest.mark.xfail
def test_transform_and_predict(tmp_path):
    assert False
