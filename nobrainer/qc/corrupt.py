"""Controlled corruption generation for QC evaluation.

Generates corrupted versions of brain MRI scans with fixed seeds
and full metadata tracking for reproducible QC benchmarking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import torch
import torchio as tio
from tqdm import tqdm

from nobrainer.qc.corruption_configs import CorruptionConfig, get_corruption_configs

logger = logging.getLogger(__name__)


def generate_corrupted_scan(
    input_path: Path,
    output_path: Path,
    config: CorruptionConfig,
    severity: int,
    seed: int,
) -> dict:
    """Apply a corruption to a single scan with fixed seed.

    Parameters:
        input_path: Path to reference NIfTI file.
        output_path: Path to save corrupted NIfTI file.
        config: Corruption configuration.
        severity: Severity level (1-5).
        seed: Random seed for reproducibility.

    Returns:
        Metadata describing the corruption applied.
    """
    torch.manual_seed(seed)

    subject = tio.Subject(t1=tio.ScalarImage(str(input_path)))
    transform = config.get_transform(severity)
    corrupted = transform(subject)

    # Save using nibabel to preserve original affine and header
    orig_nii = nib.load(str(input_path))
    corrupted_data = corrupted.t1.data.squeeze()
    # nibabel requires numpy; this is the one acceptable numpy conversion
    corrupted_nii = nib.Nifti1Image(
        corrupted_data.numpy(), orig_nii.affine, orig_nii.header
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(corrupted_nii, str(output_path))

    # Save JSON sidecar with full provenance
    metadata = {
        "ref_path": str(input_path),
        "cor_path": str(output_path),
        "corruption_type": config.name,
        "corruption_domain": config.domain,
        "severity": severity,
        "seed": seed,
        "transform_params": config.severity_params[severity],
    }
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2, default=str))

    return metadata


def generate_corrupted_dataset(
    input_dir: Path,
    output_dir: Path,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """Generate corrupted versions of all scans in input_dir.

    Parameters:
        input_dir: Directory containing reference .nii.gz files.
        output_dir: Root directory for corrupted outputs.
        corruptions: Corruption names to apply. None = all 8 types.
        severities: Severity levels to generate. None = [1, 2, 3, 4, 5].
        resume: Skip files whose output already exists.
        dry_run: Print what would be generated without writing files.

    Returns:
        Metadata dicts, one per corrupted scan.
    """
    all_configs = get_corruption_configs()
    if corruptions is not None:
        unknown = set(corruptions) - set(all_configs)
        if unknown:
            raise ValueError(f"Unknown corruptions: {unknown}")
        all_configs = {k: v for k, v in all_configs.items() if k in corruptions}
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    input_files = sorted(input_dir.glob("*.nii.gz"))
    if not input_files:
        raise FileNotFoundError(f"No .nii.gz files found in {input_dir}")

    total = len(input_files) * len(all_configs) * len(severities)
    logger.info(
        "Corruption plan: %d scans x %d types x %d severities = %d total",
        len(input_files),
        len(all_configs),
        len(severities),
        total,
    )

    if dry_run:
        logger.info("Dry run — no files will be written.")
        return []

    all_metadata: list[dict] = []
    with tqdm(total=total, desc="Generating corruptions") as pbar:
        for input_path in input_files:
            for config_name, config in all_configs.items():
                for sev in severities:
                    output_path = (
                        output_dir / config_name / f"severity_{sev}" / input_path.name
                    )
                    if resume and output_path.exists():
                        pbar.update(1)
                        continue

                    seed = hash(f"{input_path.name}_{config_name}_{sev}") % (2**32)

                    try:
                        meta = generate_corrupted_scan(
                            input_path, output_path, config, sev, seed
                        )
                        all_metadata.append(meta)
                    except Exception as exc:
                        logger.warning(
                            "Failed: %s %s sev%d: %s",
                            input_path.name,
                            config_name,
                            sev,
                            exc,
                        )
                        all_metadata.append(
                            {
                                "ref_path": str(input_path),
                                "cor_path": str(output_path),
                                "corruption_type": config_name,
                                "severity": sev,
                                "error": str(exc),
                            }
                        )
                    pbar.update(1)

    return all_metadata
