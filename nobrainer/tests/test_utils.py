import multiprocessing
import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from nobrainer.io import read_csv
import nobrainer.utils as nbutils


def test_get_data():
    csv_path = nbutils.get_data()
    assert Path(csv_path).is_file()

    files = read_csv(csv_path)
    assert len(files) == 10
    assert all(len(r) == 2 for r in files)
    for x, y in files:
        assert Path(x).is_file()
        assert Path(y).is_file()


def test_get_all_cpus():
    assert nbutils._get_all_cpus() == multiprocessing.cpu_count()
    os.environ['SLURM_CPUS_ON_NODE'] = "128"
    assert nbutils._get_all_cpus() == 128


def test_streaming_stats():
    # batch, depth, height, width, classes.
    x = np.random.rand(5, 8, 8, 8, 20)
    x[0] *= 0.2
    x[1] *= 0.5
    x[2] *= 0.8
    x[3] *= 1.1
    x = np.clip(x, 0, 1, out=x)

    s = nbutils.StreamingStats()
    for i in range(x.shape[0]):
        s.update(x[i])
    assert_allclose(s.mean, np.mean(x, axis=0))

    s.reset()
    x = [1, 2, 3, 11, 12, 13]
    for i in x:
        s.update(i)
    assert_allclose(s.mean, np.mean(x))
    assert_allclose(s.variance(), np.var(x))

    # Shape mismatch.
    with pytest.raises(ValueError):
        s.update([10, 10])
