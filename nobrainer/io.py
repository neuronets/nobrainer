# -*- coding: utf-8 -*-
"""Collection of methods for input/output."""

import csv
import json

import nibabel as nib
import numpy as np


def read_csv(filepath, header=True, delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.

    Args:
        filepath: path-like, path to CSV file.
        header: boolean, if true, skip first row.
        delimiter: str, character that separates values in the file.

    Returns:
        Nested list of CSV contents, where each sublist is one row.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if header:
            next(reader)  # skip header
        return [row for row in reader]


def read_json(filepath, **kwargs):
    """Load JSON file `filepath` as dictionary. `kwargs` are keyword arguments
    for `json.load()`.

    Args:
        filepath: path-like, path to JSON file.

    Returns:
        Dictionary of JSON contents.
    """
    with open(filepath, 'r') as fp:
        return json.load(fp, **kwargs)


def read_mapping(filepath, header=True, delimiter=','):
    """Read CSV to dictionary, where first column becomes keys and second
    columns becomes values. Keys and values must be integers.

    Args:
        filepath: path-like, path to CSV file.
        header: boolean, if true, skip first row.
        delimiter: str, character that separates values in the file.

    Return:
        Dictionary.
    """
    mapping = read_csv(filepath, header=header, delimiter=delimiter)
    return {int(row[0]): int(row[1]) for row in mapping}


def read_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data from a neuroimaging file.

    Args:
        filepath: path-like, path to volume file.
        dtype: dtype-like or str, data type of the volume data.
        return_affine: boolean, if true, return tuple of volume data and
            affine.

    Returns:
        Numpy array of volume data. If `return_affine` is true, return tuple of
        volume data and affine.
    """
    img = nib.load(filepath)
    data = np.asarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    img.uncache()
    return data if not return_affine else (data, img.affine)


def save_csv(rows, filepath, mode='w', delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.

    Args:
        rows: list of lists, nested list, where each sublist is one row.
        filepath: path-like, path to save CSV.
        mode: str, mode in which to open file object.
        delimiter: str, character to use to separate items.
    """
    with open(filepath, mode=mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerows(rows)


def save_json(obj, filepath, indent=4, **kwargs):
    """Save object as JSON file.

    Args:
        obj: object to save to JSON.
        filepath: path-like, filepath of JSON file.
        indent: int, number of spaces per indent.
        kwargs: keyword arguments for `json.dump()`.
    """
    with open(filepath, 'w') as fp:
        json.dump(obj, fp, indent=indent, **kwargs)
        fp.write('\n')
