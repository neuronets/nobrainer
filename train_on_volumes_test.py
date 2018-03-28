"""General tests for `train_on_volumes.py`."""

import os
import subprocess
import tempfile

import nibabel as nb
import numpy as np
import pandas as pd


def test_brainmask():
    volume_shape = (32, 32, 32)
    n_volumes = 20

    features = np.random.rand(n_volumes, *volume_shape).astype(np.float32)
    labels = np.random.rand(n_volumes, *volume_shape) * 100
    labels = labels.astype(np.int32)
    affine = np.eye(4)

    list_of_filepaths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save niftis and save hdf5.
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

        model_dir = os.path.join(tmpdir, 'model')
        cmd = (
            "python3 train_on_volumes.py  --n-classes=2 --model=meshnet"
            " --optimizer=Adam --learning-rate=0.001 --batch-size=2"
            " --model-dir={model_dir}"
            " --csv={csv}"
            " --eval-csv={csv}"  # Don't do this in practice...
            " --vol-shape 32 32 32"
            " --block-shape 16 16 16"
            " --strides 16 16 16"  # non-overlapping blocks
            " --brainmask"
            .format(model_dir=model_dir, csv=filepaths_path)
            .split())

        subprocess.check_output(cmd)


def test_aparcaseg():
    volume_shape = (32, 32, 32)
    n_volumes = 20

    features = np.random.rand(n_volumes, *volume_shape).astype(np.float32)

    labels = np.random.choice([0, 2, 4, 6], size=(n_volumes, *volume_shape))
    labels = labels.astype(np.int32)

    affine = np.eye(4)

    list_of_filepaths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save niftis and save hdf5.
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

        mapping_df = pd.DataFrame([(0, 0), (2, 1), (4, 2), (6, 3)])
        mapping_df.columns = ['orig', 'new']
        mapping_path = os.path.join(tmpdir, 'mapping.csv')
        mapping_df.to_csv(mapping_path, index=False)

        model_dir = os.path.join(tmpdir, 'model')
        cmd = (
            "python3 train_on_volumes.py  --n-classes=2 --model=meshnet"
            " --optimizer=Adam --learning-rate=0.001 --batch-size=2"
            " --model-dir={model_dir}"
            " --csv={csv}"
            " --eval-csv={csv}"  # Don't do this in practice...
            " --vol-shape 32 32 32"
            " --block-shape 16 16 16"
            " --strides 16 16 16"  # non-overlapping blocks
            " --aparcaseg-mapping={mapping}"
            .format(
                model_dir=model_dir, csv=filepaths_path, mapping=mapping_path)
            .split())

        subprocess.check_output(cmd)
