# -*- coding: utf-8 -*-
"""Collection of methods for input/output."""

import csv
import json

import nibabel as nib


def read_csv(filepath, header=True, delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if header:
            next(reader)  # skip header
        return [row for row in reader]


def read_json(filepath, **kwargs):
    """Load JSON file `filepath` as dictionary. `kwargs` are keyword arguments
    for `json.load()`.
    """
    with open(filepath, 'r') as fp:
        return json.load(fp, **kwargs)


def read_mapping(filepath, header=True, delimiter=','):
    """Read CSV to dictionary, where first column becomes keys and second
    columns becomes values. Keys and values must be integers.
    """
    mapping = read_csv(filepath, header=header, delimiter=delimiter)
    return {int(row[0]): int(row[1]) for row in mapping}


def read_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data from a neuroimaging file."""
    img = nib.load(filepath)
    data = img.get_fdata(caching='unchanged', dtype=dtype)
    return data if not return_affine else (data, img.affine)


def save_json(obj, filepath, indent=4, **kwargs):
    """Save `obj` to JSON file `filepath`. `kwargs` are keyword arguments for
    `json.dump()`.
    """
    with open(filepath, 'w') as fp:
        json.dump(obj, fp, indent=indent, **kwargs)
        fp.write('\n')
