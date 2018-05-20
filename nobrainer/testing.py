# -*- coding: utf-8 -*-
"""Utilities for nobrainer tests."""

import nibabel as nib
import numpy as np
import pytest

from nobrainer.io import save_csv


@pytest.fixture(scope='session')
def csv_of_volumes(tmpdir_factory):
    """Create random Nifti volumes for use in testing, and return filepath to
    CSV, which contains rows of filepaths to `(features, labels)`.
    """
    savedir = tmpdir_factory.mktemp('data')
    volume_shape = (8, 8, 8)
    n_volumes = 2

    features = np.random.rand(n_volumes, *volume_shape).astype(np.float32) * 10
    labels = np.random.rand(n_volumes, *volume_shape)
    labels = labels.astype(np.int32)
    affine = np.eye(4)
    list_of_filepaths = []

    for idx in range(n_volumes):
        fpf = str(savedir.join('{}f.nii.gz')).format(idx)
        fpl = str(savedir.join('{}l.nii.gz')).format(idx)

        nib.save(nib.Nifti1Image(features[idx], affine), fpf)
        nib.save(nib.Nifti1Image(labels[idx], affine), fpl)
        list_of_filepaths.append((fpf, fpl))

    filepath = savedir.join("features_labels.csv")
    save_csv(rows=list_of_filepaths, filepath=str(filepath))

    return str(filepath)
