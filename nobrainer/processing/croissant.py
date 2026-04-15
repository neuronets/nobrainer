"""Croissant-ML JSON-LD metadata helpers for nobrainer estimators."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


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
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
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
                "sha256": (
                    _sha256(save_dir / "model.pth")
                    if (save_dir / "model.pth").exists()
                    else ""
                ),
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


def write_checkpoint_croissant(
    checkpoint_dir: Path,
    model: Any,
    optimizer: Any,
    criterion: Any,
    history: list[dict],
) -> Path:
    """Write croissant.json alongside a training checkpoint.

    Lighter-weight than :func:`write_model_croissant` — works with the raw
    model/optimizer/criterion objects available inside :func:`~nobrainer.training.fit`
    rather than requiring an estimator wrapper.
    """
    import torch

    import nobrainer

    checkpoint_dir = Path(checkpoint_dir)

    metadata = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": f"nobrainer-{type(model).__name__}",
        "description": f"Trained {type(model).__name__} checkpoint via nobrainer",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "name": "best_model.pth",
                "contentUrl": "best_model.pth",
                "encodingFormat": "application/x-pytorch",
                "sha256": (
                    _sha256(checkpoint_dir / "best_model.pth")
                    if (checkpoint_dir / "best_model.pth").exists()
                    else ""
                ),
            }
        ],
        "nobrainer:provenance": {
            "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "nobrainer_version": nobrainer.__version__,
            "pytorch_version": torch.__version__,
            "optimizer": {
                "class": type(optimizer).__name__,
                "args": {k: str(v) for k, v in optimizer.defaults.items()},
            },
            "loss_function": type(criterion).__name__,
            "epochs_trained": len(history),
            "final_loss": (history[-1].get("loss") if history else None),
            "best_loss": (
                min(
                    (h["loss"] for h in history if h.get("loss") is not None),
                    default=None,
                )
                if history
                else None
            ),
            "model_architecture": type(model).__name__,
            "gpu_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        },
    }

    out = checkpoint_dir / "croissant.json"
    out.write_text(json.dumps(metadata, indent=2, default=str))
    return out


def write_dataset_croissant(
    output_path: str | Path,
    dataset: Any,
) -> Path:
    """Write Croissant-ML JSON-LD for a Dataset."""
    metadata = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
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
