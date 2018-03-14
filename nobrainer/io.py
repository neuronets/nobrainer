""""""

import csv
import json

import nibabel as nib
import numpy as np


def read_json(filepath, **kwargs):
    """Load JSON file `filepath` as dictionary. `kwargs` are keyword arguments
    for `json.load()`.
    """
    with open(filepath, 'r') as fp:
        return json.load(fp, **kwargs)


def read_csv(filepath, header=True, delimiter=','):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if header:
            next(reader)  # skip header
        return [row for row in reader]


def save_json(obj, filepath, indent=4, **kwargs):
    """Save `obj` to JSON file `filepath`. `kwargs` are keyword arguments for
    `json.dump()`.
    """
    with open(filepath, 'w') as fp:
        json.dump(obj, fp, indent=indent, **kwargs)
        fp.write('\n')


def load_volume(filepath, dtype=None, return_affine=False):
    """Return numpy array of data from a neuroimaging file."""
    img = nib.load(filepath)
    data = np.asarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    img.uncache()
    return data if not return_affine else (data, img.affine)
