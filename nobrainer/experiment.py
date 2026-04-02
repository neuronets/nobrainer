"""Experiment tracking: local file logger + optional Weights & Biases.

Provides a unified interface for logging training metrics.  The local
logger always works (writes JSON lines + CSV to the output directory).
W&B integration is optional and auto-detected.

Usage::

    from nobrainer.experiment import ExperimentTracker

    # Local-only (writes to output_dir/metrics.jsonl + metrics.csv)
    tracker = ExperimentTracker(output_dir="checkpoints/bvwn", config={...})

    # With W&B (if wandb is installed and WANDB_API_KEY is set)
    tracker = ExperimentTracker(
        output_dir="checkpoints/bvwn",
        config={"lr": 1e-4, "filters": 96},
        project="kwyk-reproduction",
        tags=["bvwn_multi_prior", "50-class"],
    )

    for epoch in range(epochs):
        tracker.log({"epoch": epoch, "train_loss": loss, "val_dice": dice})

    tracker.finish()
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _LocalLogger:
    """Write metrics to JSON lines + CSV in the output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._csv_writer = None
        self._csv_file = None
        self._fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any]) -> None:
        # JSON lines (append)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(metrics, default=str) + "\n")

        # CSV (create header on first call, append rows)
        if self._csv_writer is None:
            self._fieldnames = list(metrics.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

    def log_config(self, config: dict[str, Any]) -> None:
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def finish(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


class _WandbLogger:
    """Log metrics to Weights & Biases."""

    def __init__(
        self,
        config: dict[str, Any],
        project: str | None,
        name: str | None,
        tags: list[str] | None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project or "nobrainer",
            name=name,
            config=config,
            tags=tags,
            reinit=True,
        )

    def log(self, metrics: dict[str, Any]) -> None:
        self._wandb.log(metrics)

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config, allow_val_change=True)

    def finish(self) -> None:
        self._wandb.finish()


class ExperimentTracker:
    """Unified experiment tracker with local + optional W&B backends.

    The local backend always runs, writing ``metrics.jsonl``,
    ``metrics.csv``, and ``config.json`` to *output_dir*.  W&B is
    activated when:

    1. ``wandb`` is installed, AND
    2. ``WANDB_API_KEY`` is set or ``use_wandb=True`` is passed.

    Parameters
    ----------
    output_dir : str or Path
        Directory for local metric files.
    config : dict, optional
        Hyperparameters / configuration to log.
    project : str, optional
        W&B project name (default ``"nobrainer"``).
    name : str, optional
        W&B run name.
    tags : list of str, optional
        W&B run tags.
    use_wandb : bool or None
        Force W&B on/off.  None = auto-detect (use if installed + key set).
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
        project: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        use_wandb: bool | None = None,
    ) -> None:
        self._backends: list[Any] = []

        # Local logger (always active)
        local = _LocalLogger(Path(output_dir))
        self._backends.append(local)

        # Save config locally
        if config:
            local.log_config(config)

        # W&B (optional)
        if use_wandb is None:
            use_wandb = (
                os.environ.get("WANDB_API_KEY") is not None
                or os.environ.get("WANDB_MODE") == "offline"
            )
        if use_wandb:
            try:
                wb = _WandbLogger(
                    config=config or {},
                    project=project,
                    name=name,
                    tags=tags,
                )
                self._backends.append(wb)
                logger.info("W&B tracking enabled (project=%s)", project)
            except Exception as exc:
                logger.warning("W&B init failed: %s — using local only", exc)

        backend_names = [type(b).__name__ for b in self._backends]
        logger.info("Experiment tracking: %s", ", ".join(backend_names))

    def log(self, metrics: dict[str, Any]) -> None:
        """Log a dict of metrics to all backends."""
        for backend in self._backends:
            backend.log(metrics)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log/update configuration to all backends."""
        for backend in self._backends:
            backend.log_config(config)

    def finish(self) -> None:
        """Finalize all backends (flush files, end W&B run)."""
        for backend in self._backends:
            backend.finish()

    def callback(self, **extra_fields) -> callable:
        """Return a training callback that logs epoch metrics.

        The returned callable has signature ``(epoch, logs, model)`` —
        matching the callback protocol in :func:`nobrainer.training.fit`
        and :class:`Segmentation.fit`.

        Parameters
        ----------
        **extra_fields
            Extra key-value pairs included in every log entry (e.g.,
            ``variant="bvwn_multi_prior"``).

        Example::

            tracker = ExperimentTracker("checkpoints/bvwn", config={...})
            seg.fit(ds, epochs=50, callbacks=[tracker.callback(variant="ssd")])
            tracker.finish()
        """

        def _cb(epoch: int, logs: dict, model: Any) -> None:
            self.log({"epoch": epoch, **logs, **extra_fields})

        return _cb
