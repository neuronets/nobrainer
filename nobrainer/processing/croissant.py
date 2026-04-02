"""Croissant-ML JSON-LD metadata helpers for nobrainer estimators."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_checksums(dataset: Any) -> list[dict]:
    """Extract file paths and SHA256 checksums from a Dataset."""
    if dataset is None:
        return []
    checksums = []
    for item in getattr(dataset, "data", []):
        img = item.get("image", "") if isinstance(item, dict) else ""
        if img and Path(img).exists():
            checksums.append({"path": str(img), "sha256": _sha256(img)})
    return checksums


def write_model_croissant(
    save_dir: Path,
    estimator: Any,
    training_result: dict | None,
    dataset: Any,
) -> Path:
    """Write croissant.json with Croissant-ML JSON-LD metadata.

    Includes provenance (source datasets with SHA256), training parameters,
    model architecture info, and version stamps.
    """
    import torch

    import nobrainer

    result = training_result or {}

    # Extract optimizer info from estimator if available
    opt_class = getattr(estimator, "_optimizer_class", "Adam")
    opt_args = getattr(estimator, "_optimizer_args", {})
    loss_name = getattr(estimator, "_loss_name", "unknown")

    metadata = {
        "@context": {"@vocab": "http://mlcommons.org/croissant/"},
        "@type": "cr:Dataset",
        "name": f"nobrainer-{getattr(estimator, 'base_model', 'model')}",
        "description": (
            f"Trained {getattr(estimator, 'base_model', 'model')} model "
            f"via nobrainer"
        ),
        "distribution": [
            {
                "@type": "cr:FileObject",
                "name": "model.pth",
                "contentUrl": "model.pth",
                "encodingFormat": "application/x-pytorch",
            }
        ],
        "nobrainer:provenance": {
            "source_datasets": _dataset_checksums(dataset),
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "nobrainer_version": nobrainer.__version__,
            "pytorch_version": torch.__version__,
            "optimizer": {
                "class": str(opt_class),
                "args": {k: str(v) for k, v in (opt_args or {}).items()},
            },
            "loss_function": str(loss_name),
            "epochs_trained": len(result.get("history", [])),
            "final_loss": (
                result["history"][-1].get("loss") if result.get("history") else None
            ),
            "best_loss": (
                min(
                    (h["loss"] for h in result["history"] if h.get("loss") is not None),
                    default=None,
                )
                if result.get("history")
                else None
            ),
            "model_architecture": getattr(estimator, "base_model", "unknown"),
            "model_args": getattr(estimator, "model_args", None) or {},
            "n_classes": getattr(estimator, "n_classes_", None),
            "block_shape": list(getattr(estimator, "block_shape_", []) or []),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    out = save_dir / "croissant.json"
    out.write_text(json.dumps(metadata, indent=2, default=str))
    return out


def write_dataset_croissant(
    output_path: str | Path,
    dataset: Any,
) -> Path:
    """Write Croissant-ML JSON-LD for a Dataset."""
    metadata = {
        "@context": {"@vocab": "http://mlcommons.org/croissant/"},
        "@type": "cr:Dataset",
        "name": "nobrainer-dataset",
        "description": "Brain MRI dataset for nobrainer",
        "distribution": [],
        "recordSet": [],
    }

    checksums = _dataset_checksums(dataset)
    for item in checksums:
        metadata["distribution"].append(
            {
                "@type": "cr:FileObject",
                "name": Path(item["path"]).name,
                "contentUrl": item["path"],
                "sha256": item["sha256"],
            }
        )

    metadata["nobrainer:dataset_info"] = {
        "volume_shape": list(getattr(dataset, "volume_shape", []) or []),
        "n_classes": getattr(dataset, "n_classes", None),
        "block_shape": list(getattr(dataset, "_block_shape", []) or []),
        "n_volumes": len(getattr(dataset, "data", [])),
    }

    output_path = Path(output_path)
    output_path.write_text(json.dumps(metadata, indent=2, default=str))
    return output_path


def validate_croissant(path: str | Path) -> bool:
    """Validate croissant.json using mlcroissant (if installed)."""
    try:
        import mlcroissant

        mlcroissant.Dataset(jsonld=str(path))
        return True
    except ImportError:
        return True  # Skip validation if not installed
    except Exception:
        return False
