"""VLM-based quality evaluation wrapper.

Thin wrapper around VLM inference for quality scoring.
Model loading and inference logic lives in the experiment scripts
(code/08a_eval_3d_vlms.py, code/08b_eval_2d_vlms.py,
code/09_finetune_lora.py). This module provides the shared prompts and
response parsers.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

QC_PROMPT = (
    "Assess the quality of this brain MRI scan. Rate it from 1 (unusable, "
    "severe artifacts making it unsuitable for analysis) to 5 (excellent "
    "quality, no visible artifacts). Describe any quality issues you observe "
    "including: motion artifacts, noise, blurring, intensity inhomogeneity, "
    "ghosting, or resolution problems.\n"
    "Output your response in this exact format:\n"
    "SCORE: [integer 1-5]\n"
    "REASON: [one paragraph description]"
)

QC_DUAL_PROMPT = (
    "Assess this brain MRI scan on two axes:\n"
    "1. Segmentation quality (1=unusable, 5=excellent) — how well do brain "
    "structures segment under SynthSeg-style whole-brain parcellation?\n"
    "2. Cortical thickness reliability (1=highly distorted, 5=stable) — how "
    "reliable are per-region cortical thickness measurements?\n"
    "Output your response in this exact format on one line:\n"
    "Quality: [integer 1-5] Thickness: [integer 1-5]"
)


def parse_qc_response(text: str) -> dict[str, int | str | bool]:
    """Parse a VLM response into structured score and reason.

    Parameters:
        text: Raw VLM output text.

    Returns:
        Keys: "score" (int or None), "reason" (str), "parse_success" (bool).
    """
    result: dict[str, int | str | bool] = {
        "score": None,
        "reason": "",
        "parse_success": False,
    }

    # Try structured format first
    score_match = re.search(r"SCORE:\s*(\d)", text)
    if score_match:
        score = int(score_match.group(1))
        if 1 <= score <= 5:
            result["score"] = score
            result["parse_success"] = True

    # Fallback: extract any digit 1-5
    if result["score"] is None:
        digits = re.findall(r"\b([1-5])\b", text)
        if digits:
            result["score"] = int(digits[0])
            result["parse_success"] = False  # mark as fallback

    # Extract reason
    reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()
    else:
        result["reason"] = text.strip()

    return result


def parse_dual_qc_response(text: str) -> dict[str, int | None | bool]:
    """Parse dual-head QC output ``Quality: N Thickness: M`` into bucket scores.

    Used by ``code/09_finetune_lora.py`` for two-head VLM fine-tuning. The
    fine-tuned model is trained to emit a joint output string with integer
    scores 1-5 on both quality (Dice-based) and thickness (cortical-thickness-
    shift-based) axes; this parser extracts the two integers for downstream
    SRCC computation against ground-truth ``mean_dice``.

    Tolerant to: case variation (``quality`` / ``Quality`` / ``QUALITY``),
    whitespace and punctuation between key and value (``Quality: 4`` /
    ``Quality=4`` / ``Quality:4``), trailing decimals (``Quality: 4.0``
    truncates via ``int(float(...))``), and key-value separators on the
    same or different lines.

    Out-of-range scores (< 1 or > 5) parse to ``None`` rather than being
    clamped, so downstream code can distinguish "model emitted nonsense"
    from "model emitted a valid bucket".

    Parameters:
        text: Raw VLM output text.

    Returns:
        Dict with keys ``"quality"`` (int 1-5 or None), ``"thickness"``
        (int 1-5 or None), and ``"parse_success"`` (bool — True iff
        BOTH heads parsed in range).
    """
    result: dict[str, int | None | bool] = {
        "quality": None,
        "thickness": None,
        "parse_success": False,
    }

    quality_match = re.search(r"quality\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if quality_match:
        try:
            score = int(float(quality_match.group(1)))
            if 1 <= score <= 5:
                result["quality"] = score
        except ValueError:
            pass

    thickness_match = re.search(
        r"thickness\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if thickness_match:
        try:
            score = int(float(thickness_match.group(1)))
            if 1 <= score <= 5:
                result["thickness"] = score
        except ValueError:
            pass

    result["parse_success"] = (
        result["quality"] is not None and result["thickness"] is not None
    )
    return result
