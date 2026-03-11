"""Loss functions for 3-D semantic segmentation (PyTorch / MONAI)."""

from __future__ import annotations

from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
import torch

# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def dice(
    sigmoid: bool = False,
    softmax: bool = False,
    squared_pred: bool = False,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    **kwargs,
) -> DiceLoss:
    """Return a MONAI ``DiceLoss`` instance.

    Parameters
    ----------
    sigmoid : bool
        Apply sigmoid to predictions before computing Dice.
    softmax : bool
        Apply softmax to predictions before computing Dice.
    squared_pred : bool
        Use squared predictions in the denominator.
    smooth_nr, smooth_dr : float
        Numerator/denominator smoothing to avoid division by zero.
    **kwargs
        Extra keyword arguments forwarded to ``monai.losses.DiceLoss``.
    """
    return DiceLoss(
        sigmoid=sigmoid,
        softmax=softmax,
        squared_pred=squared_pred,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        **kwargs,
    )


def generalized_dice(
    sigmoid: bool = False,
    softmax: bool = False,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    **kwargs,
) -> GeneralizedDiceLoss:
    """Return a MONAI ``GeneralizedDiceLoss`` instance."""
    return GeneralizedDiceLoss(
        sigmoid=sigmoid,
        softmax=softmax,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        **kwargs,
    )


def jaccard(
    sigmoid: bool = False,
    softmax: bool = False,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    **kwargs,
) -> DiceLoss:
    """Return a Dice loss configured for Jaccard (IoU) computation.

    The Jaccard index equals ``intersection / union``; setting
    ``jaccard=True`` in MONAI's ``DiceLoss`` switches the denominator
    accordingly.
    """
    return DiceLoss(
        sigmoid=sigmoid,
        softmax=softmax,
        jaccard=True,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        **kwargs,
    )


def tversky(
    alpha: float = 0.3,
    beta: float = 0.7,
    sigmoid: bool = False,
    softmax: bool = False,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    **kwargs,
) -> TverskyLoss:
    """Return a MONAI ``TverskyLoss`` instance.

    Parameters
    ----------
    alpha : float
        Weight of false positives.
    beta : float
        Weight of false negatives.
    """
    return TverskyLoss(
        alpha=alpha,
        beta=beta,
        sigmoid=sigmoid,
        softmax=softmax,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Stubs — implemented in US2 (elbo) and US3 (wasserstein)
# ---------------------------------------------------------------------------


def elbo(
    model: torch.nn.Module,
    kl_weight: float,
    reconstruction_loss: torch.Tensor,
) -> torch.Tensor:
    """Compute ELBO = reconstruction_loss + kl_weight * KL.

    The KL term is accumulated by Pyro sampling during the forward pass
    of Bayesian modules (:class:`~nobrainer.models.bayesian.layers.BayesianConv3d`
    and :class:`~nobrainer.models.bayesian.layers.BayesianLinear`).

    Parameters
    ----------
    model : nn.Module
        A model with one or more Bayesian layers whose ``.kl`` attributes
        have been populated by a recent forward pass.
    kl_weight : float
        Scalar multiplier for the KL divergence term (often ``1 / N_data``
        or ``1 / N_batches``).
    reconstruction_loss : torch.Tensor
        Scalar reconstruction loss (e.g., Dice or cross-entropy) already
        computed for the current batch.

    Returns
    -------
    torch.Tensor
        Scalar ELBO = reconstruction_loss + kl_weight * KL.
    """
    from .models.bayesian.utils import accumulate_kl

    kl = accumulate_kl(model)
    return reconstruction_loss + kl_weight * kl


def wasserstein(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Wasserstein (Earth-Mover) loss for GAN training.

    .. note::
        Full implementation is in Phase 5 (US3 — Generative Models).
    """
    raise NotImplementedError(
        "wasserstein() is implemented in Phase 5 (US3). "
        "Import from nobrainer.models.generative after completing Phase 5."
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_losses = {
    "dice": dice,
    "generalized_dice": generalized_dice,
    "jaccard": jaccard,
    "tversky": tversky,
    "elbo": elbo,
    "wasserstein": wasserstein,
}


def get(name: str):
    """Return loss factory by name (case-insensitive)."""
    try:
        return _losses[name.lower()]
    except KeyError:
        avail = ", ".join(_losses)
        raise ValueError(f"Unknown loss '{name}'. Available: {avail}") from None
