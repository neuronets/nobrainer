"""SLURM utilities for preemptible training with checkpoint/resume.

Provides signal handling for SLURM preemption and checkpoint persistence
so training jobs can be interrupted and resumed automatically via
``--requeue``.

Usage::

    from nobrainer.slurm import (
        SlurmPreemptionHandler,
        save_checkpoint,
        load_checkpoint,
    )

    handler = SlurmPreemptionHandler()
    start_epoch, metrics = load_checkpoint(ckpt_dir, model, optimizer)

    for epoch in range(start_epoch, total_epochs):
        train_one_epoch(...)
        save_checkpoint(ckpt_dir, model, optimizer, epoch, metrics)
        if handler.preempted:
            break  # job will be requeued by SLURM
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
from typing import Any

import torch

logger = logging.getLogger(__name__)


class SlurmPreemptionHandler:
    """Handle SLURM preemption signals for graceful checkpoint-and-exit.

    SLURM sends a configurable signal (default SIGUSR1 via ``--signal``)
    before killing a preempted job.  This handler sets a flag so the
    training loop can checkpoint and exit cleanly.  The ``--requeue``
    sbatch flag then re-submits the job.

    On non-SLURM systems (no ``SLURM_JOB_ID`` environment variable),
    the handler is still safe to create but will never fire.

    Parameters
    ----------
    sig : signal.Signals
        Signal to catch (default ``SIGUSR1``).
    """

    def __init__(self, sig: signal.Signals = signal.SIGUSR1) -> None:
        self.preempted = False
        self._sig = sig
        try:
            signal.signal(sig, self._handle)
            logger.info("SLURM preemption handler registered (signal=%s)", sig.name)
        except (OSError, ValueError):
            # Signal registration can fail in non-main threads or on Windows
            logger.debug("Could not register signal handler for %s", sig)

    def _handle(self, signum: int, frame: Any) -> None:
        logger.warning(
            "Received preemption signal %d — will checkpoint and exit", signum
        )
        self.preempted = True

    @staticmethod
    def is_slurm_job() -> bool:
        """Return True if running inside a SLURM job."""
        return "SLURM_JOB_ID" in os.environ

    @staticmethod
    def slurm_info() -> dict[str, str]:
        """Return a dict of useful SLURM environment variables."""
        keys = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_PARTITION",
            "SLURM_NODELIST",
            "SLURM_NTASKS",
            "SLURM_GPUS_ON_NODE",
            "SLURM_RESTART_COUNT",
        ]
        return {k: os.environ[k] for k in keys if k in os.environ}


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save a resumable training checkpoint.

    Writes ``checkpoint.pt`` with model weights, optimizer state, epoch,
    and metrics.  Also writes ``checkpoint_meta.json`` for inspection.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory for checkpoint files.
    model : torch.nn.Module
        Model to checkpoint.
    optimizer : torch.optim.Optimizer
        Optimizer state to persist.
    epoch : int
        Completed epoch number (0-indexed).
    metrics : dict, optional
        Accumulated training metrics.

    Returns
    -------
    Path
        Path to the written ``checkpoint.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        },
        ckpt_path,
    )

    meta = {
        "epoch": epoch,
        "best_loss": (metrics or {}).get("best_loss"),
        "train_losses": (metrics or {}).get("train_losses", [])[-3:],
    }
    with open(checkpoint_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Checkpoint saved: epoch %d → %s", epoch, ckpt_path)
    return ckpt_path


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    """Load a training checkpoint and return ``(start_epoch, metrics)``.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing ``checkpoint.pt``.
    model : torch.nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer or None
        Optimizer to restore.  If None, only model is loaded.

    Returns
    -------
    start_epoch : int
        Next epoch to train (checkpoint epoch + 1).
    metrics : dict
        Accumulated metrics from previous training.
    """
    ckpt_path = Path(checkpoint_dir) / "checkpoint.pt"
    if not ckpt_path.exists():
        logger.info("No checkpoint at %s — starting from scratch", ckpt_path)
        return 0, {}

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    metrics = ckpt.get("metrics", {})
    logger.info(
        "Resumed from checkpoint: epoch %d, best_loss=%.6f",
        ckpt["epoch"],
        metrics.get("best_loss", float("inf")),
    )
    return start_epoch, metrics
