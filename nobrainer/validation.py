#!/usr/bin/env python3

from pathlib import Path

import nibabel as nib
import numpy as np

from .io import read_mapping, read_volume
from .metrics import dice as dice_numpy
from .prediction import predict as _predict
from .volume import normalize_numpy, replace

DT_X = "float32"


def validate_from_filepath(
    filepath,
    predictor,
    block_shape,
    n_classes,
    mapping_y,
    return_variance=False,
    return_entropy=False,
    return_array_from_images=False,
    n_samples=1,
    normalizer=normalize_numpy,
    batch_size=4,
):
    """Computes dice for a prediction compared to a ground truth image.

    Args:
        filepath: tuple, tuple of paths to existing neuroimaging volume (index 0)
         and ground truth (index 1).
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        n_classes: int, number of classifications the model is trained to output.
        mapping_y: path-like, path to csv mapping file per command line argument.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        return_variance: Boolean. If set True, it returns the running population
            variance along with mean. Note, if the n_samples is smaller or equal to 1,
            the variance will not be returned; instead it will return None
        return_entropy: Boolean. If set True, it returns the running entropy.
            along with mean.
        return_array_from_images: Boolean. If set True and the given input is either
            image, filepath, or filepaths, it will return arrays of [mean, variance,
            entropy] instead of images of them. Also, if the input is array, it will
            simply return array, whether or not this flag is True or False.
        n_samples: The number of sampling. If set as 1, it will just return the
            single prediction value.
        normalizer: callable, function that accepts an ndarray and returns an
            ndarray. Called before separating volume into blocks.
        batch_size: int, number of sub-volumes per batch for prediction.
        dtype: str or dtype object, dtype of features.

    Returns:
        `nibabel.spatialimages.SpatialImage` or arrays of predictions of
        mean, variance(optional), and entropy (optional).
    """
    if not Path(filepath[0]).is_file():
        raise FileNotFoundError("could not find file {}".format(filepath[0]))
    img = nib.load(filepath[0])
    y = read_volume(filepath[1], dtype=np.int32)

    outputs = _predict(
        inputs=img,
        predictor=predictor,
        block_shape=block_shape,
        return_variance=return_variance,
        return_entropy=return_entropy,
        return_array_from_images=return_array_from_images,
        n_samples=n_samples,
        normalizer=normalizer,
        batch_size=batch_size,
    )
    prediction_image = outputs[0].get_data()
    y = replace(y, read_mapping(mapping_y))
    dice = get_dice_for_images(prediction_image, y, n_classes)
    return outputs, dice


def get_dice_for_images(pred, gt, n_classes):
    """Computes dice for a prediction compared to a ground truth image.

    Args:
        pred: nibabel.spatialimages.SpatialImage, a predicted image.
        gt: nibabel.spatialimages.SpatialImage, a ground-truth image.


    Returns:
        `nibabel.spatialimages.SpatialImage`.
    """
    dice = np.zeros(n_classes)
    for i in range(n_classes):
        u = np.equal(pred, i)
        v = np.equal(gt, i)
        dice[i] = dice_numpy(u, v)

    return dice


def validate_from_filepaths(
    filepaths,
    predictor,
    block_shape,
    n_classes,
    mapping_y,
    output_path,
    return_variance=False,
    return_entropy=False,
    return_array_from_images=False,
    n_samples=1,
    normalizer=normalize_numpy,
    batch_size=4,
    dtype=DT_X,
):
    """Yield predictions from filepaths using a SavedModel.

    Args:
        test_csv: list, neuroimaging volume filepaths on which to predict.
        n_classes: int, number of classifications the model is trained to output.
        mapping_y: path-like, path to csv mapping file per command line argument.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        predictor: TensorFlow Predictor object, predictor from previously
            trained model.
        block_shape: tuple of len 3, shape of blocks on which to predict.
        normalizer: callable, function that accepts an ndarray and returns
            an ndarray. Called before separating volume into blocks.
        batch_size: int, number of sub-volumes per batch for prediction.
        dtype: str or dtype object, dtype of features.

    Returns:
        None
    """
    for filepath in filepaths:

        outputs, dice = validate_from_filepath(
            filepath=filepath,
            predictor=predictor,
            n_classes=n_classes,
            mapping_y=mapping_y,
            block_shape=block_shape,
            return_variance=return_variance,
            return_entropy=return_entropy,
            return_array_from_images=return_array_from_images,
            n_samples=n_samples,
            normalizer=normalizer,
            batch_size=batch_size,
            dtype=dtype,
        )

        outpath = Path(filepath[0])
        output_path = Path(output_path)
        suffixes = "".join(s for s in outpath.suffixes)
        mean_path = output_path / (outpath.stem + "_mean" + suffixes)
        variance_path = output_path / (outpath.stem + "_variance" + suffixes)
        entropy_path = output_path / (outpath.stem + "_entropy" + suffixes)
        dice_path = output_path / (outpath.stem + "_dice.npy")
        # if mean_path.is_file() or variance_path.is_file() or entropy_path.is_file():
        #     raise Exception(str(mean_path) + " or " + str(variance_path) +
        #                     " or " + str(entropy_path) + " already exists.")

        nib.save(outputs[0], mean_path.as_posix())  # fix
        if not return_array_from_images:
            include_variance = (n_samples > 1) and (return_variance)
            include_entropy = (n_samples > 1) and (return_entropy)
            if include_variance and return_entropy:
                nib.save(outputs[1], str(variance_path))
                nib.save(outputs[2], str(entropy_path))
            elif include_variance:
                nib.save(outputs[1], str(variance_path))
            elif include_entropy:
                nib.save(outputs[1], str(entropy_path))

        print(filepath[0])
        print("Dice: " + str(np.mean(dice)))
        np.save(dice_path, dice)
