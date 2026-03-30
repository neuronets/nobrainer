"""Shared utilities for kwyk reproduction experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import signal
from typing import Any

import numpy as np
import torch


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


# ---------------------------------------------------------------------------
# Checkpoint / resume for SLURM preemptible jobs
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


class SlurmPreemptionHandler:
    """Handle SLURM preemption signals for graceful checkpoint-and-exit.

    SLURM sends SIGUSR1 (or the signal specified by ``--signal``) before
    killing a preempted job.  This handler sets a flag so the training
    loop can checkpoint and exit cleanly.  The ``--requeue`` sbatch flag
    then re-submits the job, and the training resumes from the checkpoint.

    Usage::

        handler = SlurmPreemptionHandler()
        for epoch in range(start_epoch, total_epochs):
            train_one_epoch(...)
            save_checkpoint(...)
            if handler.preempted:
                log.info("Preempted — exiting for requeue")
                sys.exit(0)
    """

    def __init__(self, sig: int = signal.SIGUSR1) -> None:
        self.preempted = False
        self._sig = sig
        signal.signal(sig, self._handle)
        _logger.info("SLURM preemption handler registered (signal=%s)", sig.name)

    def _handle(self, signum: int, frame: Any) -> None:
        _logger.warning(
            "Received preemption signal %d — will checkpoint and exit", signum
        )
        self.preempted = True


def save_training_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> Path:
    """Save a resumable training checkpoint.

    Writes ``checkpoint.pt`` containing model weights, optimizer state,
    epoch number, and accumulated metrics (losses, Dice scores, etc.).
    Also writes ``checkpoint_meta.json`` with human-readable status.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory to save checkpoint files.
    model : torch.nn.Module
        Model to checkpoint.
    optimizer : torch.optim.Optimizer
        Optimizer to checkpoint (includes momentum, lr schedule state).
    epoch : int
        Completed epoch number (0-indexed).
    metrics : dict
        Accumulated training metrics to persist across restarts.

    Returns
    -------
    Path
        Path to the written checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        ckpt_path,
    )

    # Human-readable metadata
    meta = {
        "epoch": epoch,
        "best_loss": metrics.get("best_loss", None),
        "train_losses": metrics.get("train_losses", [])[-3:],
    }
    with open(checkpoint_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    _logger.info("Checkpoint saved: epoch %d → %s", epoch, ckpt_path)
    return ckpt_path


def load_training_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    """Load a training checkpoint and return (start_epoch, metrics).

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing ``checkpoint.pt``.
    model : torch.nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer or None
        Optimizer to restore state into.  If None, only model is loaded.

    Returns
    -------
    start_epoch : int
        The next epoch to train (checkpoint epoch + 1).
    metrics : dict
        Accumulated metrics from previous training.
    """
    ckpt_path = checkpoint_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        _logger.info("No checkpoint found at %s — starting from scratch", ckpt_path)
        return 0, {}

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    metrics = ckpt.get("metrics", {})
    _logger.info(
        "Resumed from checkpoint: epoch %d, best_loss=%.6f",
        ckpt["epoch"],
        metrics.get("best_loss", float("inf")),
    )
    return start_epoch, metrics
