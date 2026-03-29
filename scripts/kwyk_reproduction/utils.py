"""Shared utilities for kwyk reproduction experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    import yaml

    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(name: str) -> logging.Logger:
    """Configure and return a logger with timestamped format.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_figure(fig: Any, path: str | Path) -> None:
    """Save a matplotlib figure, creating parent directories if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)


def compute_dice(pred: np.ndarray, label: np.ndarray) -> float:
    """Compute the Dice score between two binary volumes.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction array.
    label : np.ndarray
        Binary ground-truth array.

    Returns
    -------
    float
        Dice coefficient in [0, 1]. Returns 1.0 when both arrays are empty.
    """
    pred = pred.astype(bool)
    label = label.astype(bool)
    intersection = np.logical_and(pred, label).sum()
    total = pred.sum() + label.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def apply_label_mapping(
    label_vol: np.ndarray, mapping_csv: str | Path | None = None
) -> np.ndarray:
    """Remap FreeSurfer label codes in a volume.

    When *mapping_csv* is ``None`` the volume is binarised
    (``(vol > 0).astype(int)``).  Otherwise a CSV with columns
    ``original,new`` is loaded and used to build a lookup table that maps
    each original code to its new value.

    Parameters
    ----------
    label_vol : np.ndarray
        Integer label volume.
    mapping_csv : str, Path, or None
        Path to a CSV mapping file.  If ``None``, perform binary
        thresholding.

    Returns
    -------
    np.ndarray
        Remapped label volume with the same shape as the input.
    """
    if mapping_csv is None:
        return (label_vol > 0).astype(int)

    import csv

    mapping_csv = Path(mapping_csv)
    lookup: dict[int, int] = {}
    with open(mapping_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[int(row["original"])] = int(row["new"])

    mapper = np.vectorize(lambda v: lookup.get(v, 0))
    return mapper(label_vol)
