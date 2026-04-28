"""Quality control module for brain MRI.

Provides:
- Severity-calibrated corruption generation for QC benchmarking
- Signal-based image quality metric (IQM) extraction
- Machine preference scoring via downstream task degradation
- 3D → 2D slice extraction strategies for VLM evaluation
- Pipeline gating logic (accept/reject/review)
"""

from nobrainer.qc.corrupt import generate_corrupted_dataset, generate_corrupted_scan
from nobrainer.qc.corruption_configs import CorruptionConfig, get_corruption_configs
from nobrainer.qc.evaluate import QC_PROMPT, parse_qc_response
from nobrainer.qc.gate import QCGate
from nobrainer.qc.metrics import extract_iqms
from nobrainer.qc.preference import compute_dice_preference
from nobrainer.qc.slice_extractor import extract_slices

__all__ = [
    "CorruptionConfig",
    "get_corruption_configs",
    "generate_corrupted_scan",
    "generate_corrupted_dataset",
    "extract_iqms",
    "compute_dice_preference",
    "extract_slices",
    "QC_PROMPT",
    "parse_qc_response",
    "QCGate",
]
