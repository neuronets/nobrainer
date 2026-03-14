"""Evaluation metrics for 3-D semantic segmentation (PyTorch / MONAI)."""

from __future__ import annotations

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU

# ---------------------------------------------------------------------------
# Factory functions returning configured MONAI metric objects
# ---------------------------------------------------------------------------


def dice_metric(
    include_background: bool = True,
    reduction: str = "mean",
    **kwargs,
) -> DiceMetric:
    """Return a MONAI ``DiceMetric`` instance.

    Parameters
    ----------
    include_background : bool
        Include the background class in the Dice computation.
    reduction : str
        Reduction applied over the batch (``"mean"``, ``"sum"``, ``"none"``).
    """
    return DiceMetric(
        include_background=include_background,
        reduction=reduction,
        **kwargs,
    )


def generalized_dice_metric(
    include_background: bool = True,
    reduction: str = "mean",
    **kwargs,
) -> DiceMetric:
    """Return a ``DiceMetric`` configured for multi-class (generalised) Dice.

    MONAI's ``DiceMetric`` computes per-class Dice and averages over
    classes, which is equivalent to Generalized Dice when class weights
    are uniform.
    """
    return DiceMetric(
        include_background=include_background,
        reduction=reduction,
        **kwargs,
    )


def jaccard_metric(
    include_background: bool = True,
    reduction: str = "mean",
    **kwargs,
) -> MeanIoU:
    """Return a MONAI ``MeanIoU`` (Jaccard) metric instance."""
    return MeanIoU(
        include_background=include_background,
        reduction=reduction,
        **kwargs,
    )


def tversky_metric(
    include_background: bool = True,
    reduction: str = "mean",
    **kwargs,
) -> DiceMetric:
    """Return a ``DiceMetric`` used as a Tversky metric proxy.

    Tversky with alpha=beta=0.5 equals Dice.  For asymmetric Tversky,
    compute the Tversky index manually and wrap it in a custom metric.
    """
    return DiceMetric(
        include_background=include_background,
        reduction=reduction,
        **kwargs,
    )


def hausdorff_metric(
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = 95.0,
    directed: bool = False,
    **kwargs,
) -> HausdorffDistanceMetric:
    """Return a MONAI ``HausdorffDistanceMetric`` instance.

    Parameters
    ----------
    include_background : bool
        Include background class in distance computation.
    distance_metric : str
        ``"euclidean"``, ``"chessboard"``, or ``"taxicab"``.
    percentile : float or None
        If set, computes the *n*-th percentile Hausdorff distance (e.g.
        95 for HD95).  ``None`` returns the maximum (HD100).
    directed : bool
        Compute directed (asymmetric) Hausdorff distance.
    """
    return HausdorffDistanceMetric(
        include_background=include_background,
        distance_metric=distance_metric,
        percentile=percentile,
        directed=directed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_metrics = {
    "dice": dice_metric,
    "generalized_dice": generalized_dice_metric,
    "jaccard": jaccard_metric,
    "tversky": tversky_metric,
    "hausdorff": hausdorff_metric,
}


def get(name: str):
    """Return metric factory by name (case-insensitive)."""
    try:
        return _metrics[name.lower()]
    except KeyError:
        avail = ", ".join(_metrics)
        raise ValueError(f"Unknown metric '{name}'. Available: {avail}") from None
