"""VLM-based quality evaluation wrapper.

Thin wrapper around VLM inference for quality scoring.
Model loading and inference logic lives in the experiment scripts
(code/08a_eval_3d_vlms.py, code/08b_eval_2d_vlms.py). This module
provides the shared prompt and response parsing.
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
