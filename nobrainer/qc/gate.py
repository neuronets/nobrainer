"""Pipeline gating logic for automated QC decisions.

Given a quality score (from VLM or IQMs), decide whether a scan
should be accepted, rejected, or flagged for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Result of a QC gating decision.

    Parameters:
        action: "accept", "reject", or "review".
        score: The quality score that triggered the decision.
        reason: Human-readable explanation.
    """

    action: str
    score: float
    reason: str


class QCGate:
    """Threshold-based QC gating.

    Parameters:
        accept_threshold: Scores >= this are accepted.
        reject_threshold: Scores <= this are rejected.
        score_range: Expected (min, max) for scores. Default (1.0, 5.0)
            for VLM scores.
    """

    def __init__(
        self,
        accept_threshold: float = 3.5,
        reject_threshold: float = 2.0,
        score_range: tuple[float, float] = (1.0, 5.0),
    ) -> None:
        if reject_threshold >= accept_threshold:
            raise ValueError(
                f"reject_threshold ({reject_threshold}) must be < "
                f"accept_threshold ({accept_threshold})"
            )
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.score_range = score_range

    def decide(self, score: float) -> GateDecision:
        """Make a gating decision for a single scan.

        Parameters:
            score: Quality score.

        Returns:
            The accept/reject/review decision.
        """
        if score != score:  # NaN check
            return GateDecision(
                action="review",
                score=score,
                reason="Score is NaN; manual review required.",
            )

        if score >= self.accept_threshold:
            return GateDecision(
                action="accept",
                score=score,
                reason=(
                    f"Score {score:.2f} >= " f"accept threshold {self.accept_threshold}"
                ),
            )

        if score <= self.reject_threshold:
            return GateDecision(
                action="reject",
                score=score,
                reason=(
                    f"Score {score:.2f} <= " f"reject threshold {self.reject_threshold}"
                ),
            )

        return GateDecision(
            action="review",
            score=score,
            reason=(
                f"Score {score:.2f} between reject ({self.reject_threshold}) "
                f"and accept ({self.accept_threshold}) thresholds"
            ),
        )
