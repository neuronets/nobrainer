"""General tests for command-line usage of `vols2hdf5.py`"""

import os
import subprocess
import tempfile

import h5py
import nibabel as nb
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from nobrainer.preprocessing import as_blocks


def test_vols2hdf5_one():
    volume_shape = (32, 32, 32)
    n_volumes = 20

    features = np.random.rand(n_volumes, *volume_shape).astype(np.float32)
    labels = np.random.rand(n_volumes, *volume_shape) * 100
    labels = labels.astype(np.int32)
    affine = np.eye(4)

    list_of_filepaths = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # Save niftis.
        for idx in range(n_volumes):
            _fpf = os.path.join(tmpdir, '{}f.nii.gz'.format(idx))
            _fpl = os.path.join(tmpdir, '{}l.nii.gz'.format(idx))
            nb.save(nb.Nifti1Image(features[idx], affine), _fpf)
            nb.save(nb.Nifti1Image(labels[idx], affine), _fpl)

            list_of_filepaths.append((_fpf, _fpl))

        # Save filepaths as CSV.
        filepaths_path = os.path.join(tmpdir, 'paths.csv')
        pd.DataFrame(list_of_filepaths).to_csv(
            filepaths_path, index=False, header=True)

        # Create HDF5 based on filepaths in CSV.
        hdf5path = os.path.join(tmpdir, "data.h5")
        subprocess.check_output(
            "python3 vols2hdf5.py -o {outfile}"
            " --block-shape 16 16 16 -fdt float32 -ldt int32"
            " --chunksize 10 --ncpu 2"
            " --compression gzip --compression-opts 1"
            " {infile}".format(outfile=hdf5path, infile=filepaths_path).split()
        )

        with h5py.File(hdf5path, mode='r') as fp:
            features_16 = fp['/16x16x16/features'][:]
            labels_16 = fp['/16x16x16/labels'][:]

        features_blocked = np.concatenate(
            tuple(as_blocks(features[i], block_shape=(16, 16, 16))
                  for i in range(features.shape[0])))

        labels_blocked = np.concatenate(
            tuple(as_blocks(labels[i], block_shape=(16, 16, 16))
                  for i in range(labels.shape[0])))

        assert_array_equal(features_16, features_blocked)
        assert_array_equal(labels_16, labels_blocked)


def test_vols2hdf5_two():
    volume_shape = (64, 64, 64)
    n_volumes = 4

    features = np.random.rand(n_volumes, *volume_shape) * 100
    features = features.astype(np.int32)
    labels = np.random.rand(n_volumes, *volume_shape).astype(np.float32)
    affine = np.eye(4)

    list_of_filepaths = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # Save niftis.
        for idx in range(n_volumes):
            _fpf = os.path.join(tmpdir, '{}f.nii.gz'.format(idx))
            _fpl = os.path.join(tmpdir, '{}l.nii.gz'.format(idx))
            nb.save(nb.Nifti1Image(features[idx], affine), _fpf)
            nb.save(nb.Nifti1Image(labels[idx], affine), _fpl)

            list_of_filepaths.append((_fpf, _fpl))

        # Save filepaths as CSV.
        filepaths_path = os.path.join(tmpdir, 'paths.csv')
        pd.DataFrame(list_of_filepaths).to_csv(
            filepaths_path, index=False, header=True)

        # Create HDF5 based on filepaths in CSV.
        hdf5path = os.path.join(tmpdir, "data.h5")
        otherpath = os.path.join(tmpdir, "foundfiles.csv")
        subprocess.check_output(
            "python3 vols2hdf5.py -o {outfile}"
            " --block-shape 32 32 32"
            " --block-shape 16 16 16"
            " -fdt int32 -ldt float32"
            " --chunksize 3 --ncpu 2"
            " --compression gzip --compression-opts 1"
            " --save-filepaths {otherpath}"
            " {infile}"
            .format(
                outfile=hdf5path, infile=filepaths_path, otherpath=otherpath)
            .split())

        with h5py.File(hdf5path, mode='r') as fp:
            features_32 = fp['/32x32x32/features'][:]
            labels_32 = fp['/32x32x32/labels'][:]
            features_16 = fp['/16x16x16/features'][:]
            labels_16 = fp['/16x16x16/labels'][:]

        features_blocked_32 = np.concatenate(
            tuple(as_blocks(features[i], block_shape=(32, 32, 32))
                  for i in range(features.shape[0])))

        labels_blocked_32 = np.concatenate(
            tuple(as_blocks(labels[i], block_shape=(32, 32, 32))
                  for i in range(labels.shape[0])))

        features_blocked_16 = np.concatenate(
            tuple(as_blocks(features[i], block_shape=(16, 16, 16))
                  for i in range(features.shape[0])))

        labels_blocked_16 = np.concatenate(
            tuple(as_blocks(labels[i], block_shape=(16, 16, 16))
                  for i in range(labels.shape[0])))

        assert_array_equal(features_32, features_blocked_32)
        assert_array_equal(labels_32, labels_blocked_32)

        assert_array_equal(features_16, features_blocked_16)
        assert_array_equal(labels_16, labels_blocked_16)


def test_vols2hdf5_three():
    volume_shape = (32, 32, 32)
    n_volumes = 20

    features = np.random.rand(n_volumes, *volume_shape)
    labels = np.random.rand(n_volumes, *volume_shape) * 100
    labels = labels.astype(np.int32)
    affine = np.eye(4)

    list_of_filepaths = []

    with tempfile.TemporaryDirectory() as tmpdir:

        # Save niftis.
        for idx in range(n_volumes):
            _fpf = os.path.join(tmpdir, '{}f.nii.gz'.format(idx))
            _fpl = os.path.join(tmpdir, '{}l.nii.gz'.format(idx))
            nb.save(nb.Nifti1Image(features[idx], affine), _fpf)
            nb.save(nb.Nifti1Image(labels[idx], affine), _fpl)

            list_of_filepaths.append((_fpf, _fpl))

        # Save filepaths as CSV.
        filepaths_path = os.path.join(tmpdir, 'paths.csv')
        pd.DataFrame(list_of_filepaths).to_csv(
            filepaths_path, index=False, header=True)

        # Create HDF5 based on filepaths in CSV.
        hdf5path = os.path.join(tmpdir, "data.h5")
        subprocess.check_output(
            "python3 vols2hdf5.py -o {outfile}"
            " --block-shape 16 16 16 -fdt float32 -ldt int32"
            " --normalize zscore"
            " --chunksize 10 --ncpu 2"
            " --compression gzip --compression-opts 1"
            " {infile}".format(outfile=hdf5path, infile=filepaths_path).split()
        )

        with h5py.File(hdf5path, mode='r') as fp:
            features_16 = fp['/16x16x16/features'][:]
            labels_16 = fp['/16x16x16/labels'][:]

        for ii in range(n_volumes):
            features[ii] = (
                (features[ii] - features[ii].mean()) / features[ii].std())
        features = features.astype(np.float32)

        features_blocked = np.concatenate(
            tuple(as_blocks(features[i], block_shape=(16, 16, 16))
                  for i in range(features.shape[0])))

        labels_blocked = np.concatenate(
            tuple(as_blocks(labels[i], block_shape=(16, 16, 16))
                  for i in range(labels.shape[0])))

        assert_array_equal(features_16, features_blocked)
        assert_array_equal(labels_16, labels_blocked)
