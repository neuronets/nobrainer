"""Autoresearch loop for nobrainer.

Proposes hyperparameter diffs via the Anthropic API, applies them to a
training script, runs the experiment subprocess, and keeps improvements.

If the Anthropic API is unavailable (no key or import error) the loop
falls back to a random perturbation from a pre-defined search grid.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_GRID: dict[str, list[Any]] = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "batch_size": [2, 4, 8],
    "n_epochs": [10, 20, 50],
    "dropout_rate": [0.0, 0.1, 0.25, 0.5],
}


@dataclass
class ExperimentResult:
    """Structured record for one autoresearch experiment."""

    run_id: int
    config: dict[str, Any]
    val_dice: float | None
    outcome: str  # "improved", "degraded", "failed"
    failure_reason: str | None = None
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def run_loop(
    working_dir: str | Path,
    model_family: str = "bayesian_vnet",
    max_experiments: int = 10,
    budget_hours: float = 8.0,
    train_script: str = "train.py",
    val_dice_file: str = "val_dice.json",
    budget_timeout_per_run: float = 3600.0,
) -> list[ExperimentResult]:
    """Run the autoresearch experiment loop.

    Parameters
    ----------
    working_dir : path
        Directory containing the training script and where results are saved.
    model_family : str
        Model family name (e.g. ``"bayesian_vnet"``).
    max_experiments : int
        Maximum number of experiments to run.
    budget_hours : float
        Wall-clock budget in hours (loop stops when exceeded).
    train_script : str
        Filename of the training script relative to ``working_dir``.
    val_dice_file : str
        Filename of the validation Dice JSON written by the training script.
    budget_timeout_per_run : float
        Per-experiment subprocess timeout in seconds.

    Returns
    -------
    list[ExperimentResult]
        All experiment records (including failures).
    """
    working_dir = Path(working_dir)
    train_path = working_dir / train_script
    val_dice_path = working_dir / val_dice_file
    backup_path = working_dir / f"{train_script}.backup"
    budget_end = time.time() + budget_hours * 3600.0

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training script not found: {train_path}. "
            "Create it or copy from nobrainer.research.templates."
        )

    # Read initial config from train_script (look for JSON comment block)
    current_config = _parse_config_comment(train_path)
    best_dice: float | None = None
    results: list[ExperimentResult] = []

    logger.info("Starting autoresearch loop for %s", model_family)
    logger.info("max_experiments=%d, budget_hours=%.1f", max_experiments, budget_hours)

    for run_id in range(max_experiments):
        if time.time() >= budget_end:
            logger.info("Budget exhausted — stopping at experiment %d", run_id)
            break

        # Propose new config
        new_config = _propose_config(current_config, model_family, run_id, best_dice)
        logger.info("Experiment %d config: %s", run_id, new_config)

        # Backup train script, patch config
        shutil.copy2(train_path, backup_path)
        _patch_config(train_path, new_config)

        # Run experiment subprocess
        t0 = time.time()
        failure_reason: str | None = None
        val_dice: float | None = None
        outcome = "failed"

        try:
            proc = subprocess.run(
                ["python", str(train_path)],
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=budget_timeout_per_run,
            )
            elapsed = time.time() - t0

            # Check for failure signals
            if proc.returncode != 0:
                failure_reason = _classify_failure(proc.stderr)
            elif _has_nan(proc.stdout):
                failure_reason = "NaN in loss"
            else:
                # Read val_dice.json
                val_dice = _read_val_dice(val_dice_path)
                if val_dice is not None:
                    if best_dice is None or val_dice > best_dice:
                        outcome = "improved"
                        best_dice = val_dice
                        current_config = new_config
                    else:
                        outcome = "degraded"
                else:
                    failure_reason = "val_dice.json missing or invalid"

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            failure_reason = f"timeout after {budget_timeout_per_run:.0f}s"

        if failure_reason is not None:
            logger.warning("Experiment %d failed: %s", run_id, failure_reason)
            # Revert train script
            shutil.copy2(backup_path, train_path)

        results.append(
            ExperimentResult(
                run_id=run_id,
                config=copy.deepcopy(new_config),
                val_dice=val_dice,
                outcome=outcome,
                failure_reason=failure_reason,
                elapsed_seconds=elapsed if "elapsed" in dir() else 0.0,
            )
        )

    # Write run summary
    _write_summary(working_dir, results, model_family, best_dice)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _propose_config(
    current: dict[str, Any],
    model_family: str,
    run_id: int,
    best_dice: float | None,
) -> dict[str, Any]:
    """Propose a new config via Anthropic API or random grid search."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _propose_via_llm(current, model_family, run_id, best_dice, api_key)
        except Exception as exc:
            logger.warning(
                "Anthropic API proposal failed (%s) — falling back to random grid", exc
            )
    return _propose_random(current)


def _propose_via_llm(
    current: dict[str, Any],
    model_family: str,
    run_id: int,
    best_dice: float | None,
    api_key: str,
) -> dict[str, Any]:
    """Use Anthropic claude-sonnet-4-6 to propose a new config diff."""
    import anthropic  # type: ignore[import-untyped]

    client = anthropic.Anthropic(api_key=api_key)
    context = (
        f"You are an ML research assistant. The current training config is:\n"
        f"{json.dumps(current, indent=2)}\n\n"
        f"Model family: {model_family}\n"
        f"Experiment number: {run_id}\n"
        f"Best val_dice so far: {best_dice}\n\n"
        f"Propose a new configuration as a JSON object with updated hyperparameters "
        f"(use the same keys). Only return the JSON object, no other text."
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": context}],
    )
    raw = message.content[0].text.strip()
    # Extract JSON from the response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("LLM did not return a JSON object")
    proposed = json.loads(raw[start:end])
    # Merge with current (keep unchanged keys)
    merged = dict(current)
    merged.update(proposed)
    return merged


def _propose_random(current: dict[str, Any]) -> dict[str, Any]:
    """Random perturbation from the search grid (LLM fallback)."""
    import random

    proposed = dict(current)
    for key, values in _DEFAULT_SEARCH_GRID.items():
        if key in current:
            proposed[key] = random.choice(values)
    logger.info("Random grid proposal: %s", proposed)
    return proposed


def _parse_config_comment(path: Path) -> dict[str, Any]:
    """Extract a JSON block from a ``# CONFIG: {...}`` comment in the script."""
    with path.open() as fh:
        for line in fh:
            if line.strip().startswith("# CONFIG:"):
                try:
                    return json.loads(line.split("# CONFIG:", 1)[1].strip())
                except json.JSONDecodeError:
                    pass
    return {}


def _patch_config(path: Path, config: dict[str, Any]) -> None:
    """Replace the ``# CONFIG: {...}`` comment line in the training script."""
    lines = path.read_text().splitlines(keepends=True)
    patched = []
    found = False
    for line in lines:
        if line.strip().startswith("# CONFIG:"):
            patched.append(f"# CONFIG: {json.dumps(config)}\n")
            found = True
        else:
            patched.append(line)
    if not found:
        patched.insert(0, f"# CONFIG: {json.dumps(config)}\n")
    path.write_text("".join(patched))


def _read_val_dice(path: Path) -> float | None:
    """Read the ``val_dice`` value from a JSON file."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data.get("val_dice", data.get("dice", 0.0)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _has_nan(text: str) -> bool:
    return "nan" in text.lower() or "NaN" in text


def _classify_failure(stderr: str) -> str:
    lower = stderr.lower()
    if "out of memory" in lower or "outofmemoryerror" in lower:
        return "CUDA OOM"
    if "nan" in lower:
        return "NaN in loss"
    return "non-zero exit code"


def _write_summary(
    working_dir: Path,
    results: list[ExperimentResult],
    model_family: str,
    best_dice: float | None,
) -> None:
    """Write ``run_summary.md`` to ``working_dir``."""
    lines = [
        f"# Autoresearch Run Summary: {model_family}",
        "",
        f"Total experiments: {len(results)}",
        (
            f"Best val_dice: {best_dice:.4f}"
            if best_dice is not None
            else "Best val_dice: N/A"
        ),
        "",
        "## Experiment Log",
        "",
        "| run_id | val_dice | outcome | failure_reason | elapsed_s |",
        "|--------|----------|---------|----------------|-----------|",
    ]
    for r in results:
        dice_str = f"{r.val_dice:.4f}" if r.val_dice is not None else "—"
        lines.append(
            f"| {r.run_id} | {dice_str} | {r.outcome} | "
            f"{r.failure_reason or '—'} | {r.elapsed_seconds:.1f} |"
        )
    (working_dir / "run_summary.md").write_text("\n".join(lines) + "\n")


def commit_best_model(
    best_model_path: str | Path,
    best_config_path: str | Path,
    trained_models_path: str | Path,
    model_family: str,
    val_dice: float,
    source_run_id: str = "",
) -> dict[str, Any]:
    """Version the best model with DataLad and push to OSF.

    Parameters
    ----------
    best_model_path : path
        Path to the ``best_model.pth`` file.
    best_config_path : path
        Path to the ``best_config.json`` file.
    trained_models_path : path
        Root of the DataLad-managed ``trained_models`` dataset.
    model_family : str
        Model family name (used as subdirectory).
    val_dice : float
        Validation Dice score of the best model.
    source_run_id : str
        Run ID string for traceability.

    Returns
    -------
    dict
        ``ModelVersion`` with ``path``, ``datalad_commit``, and metadata.
    """
    import datetime

    import torch

    try:
        import datalad.api as dl  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "datalad is required for model versioning. "
            "Install it with: pip install nobrainer[versioning]"
        ) from exc

    date_str = datetime.date.today().isoformat()
    dest = (
        Path(trained_models_path)
        / "neuronets"
        / "autoresearch"
        / model_family
        / date_str
    )
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_model_path, dest / "model.pth")
    shutil.copy2(best_config_path, dest / "config.json")

    # Generate model card
    import platform

    import monai
    import pyro

    card_lines = [
        f"# Model Card: {model_family}",
        "",
        "## Architecture",
        f"- Model family: {model_family}",
        "- Framework: PyTorch",
        "",
        "## Performance",
        f"- val_dice: {val_dice:.4f}",
        f"- source_run_id: {source_run_id}",
        "",
        "## Environment",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- MONAI: {monai.__version__}",
        f"- Pyro-ppl: {pyro.__version__}",
        f"- Date: {date_str}",
    ]
    (dest / "model_card.md").write_text("\n".join(card_lines) + "\n")

    commit_msg = (
        f"autoresearch: add {model_family} model ({date_str}) val_dice={val_dice:.4f}"
    )
    dl.save(path=str(trained_models_path), message=commit_msg)

    try:
        dl.push(dataset=str(trained_models_path), to="osf")
        osf_url = "osf://"
    except Exception:
        osf_url = None

    return {
        "path": str(dest),
        "datalad_commit": commit_msg,
        "val_dice": val_dice,
        "model_family": model_family,
        "date": date_str,
        "osf_url": osf_url,
    }
