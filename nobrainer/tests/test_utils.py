import multiprocessing
import os
from pathlib import Path

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
