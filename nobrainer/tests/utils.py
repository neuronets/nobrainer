import csv

import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.stats import entropy

from ..utils import StreamingStats


@pytest.fixture(scope="session")
def csv_of_volumes(tmpdir_factory):
    """Create random Nifti volumes for use in testing, and return filepath to
    CSV, which contains rows of filepaths to `(features, labels)`.
    """
    savedir = tmpdir_factory.mktemp("data")
    volume_shape = (8, 8, 8)
    n_volumes = 100

    features = np.random.rand(n_volumes, *volume_shape).astype(np.float32) * 10
    labels = np.random.randint(0, 1, size=(n_volumes, *volume_shape))
    labels = labels.astype(np.int32)
    affine = np.eye(4)
    list_of_filepaths = []

    for idx in range(n_volumes):
        fpf = str(savedir.join("{}f.nii.gz")).format(idx)
        fpl = str(savedir.join("{}l.nii.gz")).format(idx)

        nib.save(nib.Nifti1Image(features[idx], affine), fpf)
        nib.save(nib.Nifti1Image(labels[idx], affine), fpl)
        list_of_filepaths.append((fpf, fpl))

    filepath = savedir.join("features_labels.csv")
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(list_of_filepaths)

    return str(filepath)


def test_stream_stat():
    s1 = np.array([[0.5, 0.1, 0.4]])
    s2 = np.array([[0.5, 0.2, 0.3]])
    s3 = np.array([[0.4, 0.2, 0.4]])

    st = np.concatenate((s1, s2, s3), axis=0)

    s = StreamingStats()
    s.update(s1).update(s2).update(s3)

    assert_array_equal(s.mean(), np.mean(st, axis=0))
    assert_array_equal(s.var(), np.var(st, axis=0))
    assert_array_equal(s.std(), np.std(st, axis=0))
    assert_array_equal(np.sum(s.entropy()), entropy(np.mean(st, axis=0), axis=0))
