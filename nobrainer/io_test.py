# -*- coding: utf-8 -*-
"""Tests for `nobrainer.io`."""

import numpy as np

from nobrainer.io import read_csv
from nobrainer.io import read_json
from nobrainer.io import read_mapping
from nobrainer.io import read_volume
from nobrainer.io import save_csv
from nobrainer.io import save_json
from nobrainer.testing import csv_of_volumes


def test_save_read_csv(tmpdir):
    rows = [
        ["col1", "col2"],
        ["foo", "bar"],
        ["foo", "bar"],
        ["foo", "bar"],
        ["foo", "bar"],
        ["foo", "bar"]]

    filepath = tmpdir.join("file.csv")
    save_csv(rows=rows, filepath=str(filepath))
    rows_withoutheader = read_csv(filepath=str(filepath), header=True)
    rows_withheader = read_csv(filepath=str(filepath), header=False)

    assert rows == rows_withheader
    assert rows[1:] == rows_withoutheader


def test_save_read_json(tmpdir):
    d = {
        "foo": "bar",
        "baz": "boo"}

    filepath = tmpdir.join("file.json")
    save_json(obj=d, filepath=str(filepath))
    d_read = read_json(str(filepath))
    assert d == d_read, "saved and read dictionaries not equal"


def test_read_mapping(tmpdir):
    rows = [
        ["orig", "new"],
        ["0", "1"],
        ["1", "2"],
        ["2", "3"]]

    filepath = tmpdir.join("mapping.csv")
    save_csv(rows=rows, filepath=str(filepath))
    mapping = read_mapping(str(filepath), header=True)
    assert mapping == {0: 1, 1: 2, 2: 3}


def test_read_volume(csv_of_volumes):
    filepath = read_csv(csv_of_volumes)[0][0]
    volume = read_volume(filepath, dtype='float32', return_affine=False)
    assert volume.sum()
    assert volume.shape == (8, 8, 8)
    assert volume.dtype == np.float32

    volume, affine = read_volume(filepath, dtype='int32', return_affine=True)
    assert volume.sum()
    assert volume.shape == (8, 8, 8)
    assert volume.dtype == np.int32
    assert affine.shape == (4, 4)
    assert affine.sum() == 4
