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
    """Wasserstein critic loss: ``E[D(fake)] - E[D(real)]``.

    Parameters
    ----------
    y_true : torch.Tensor
        Critic scores for real samples, shape ``(N,)`` or ``(N, 1)``.
    y_pred : torch.Tensor
        Critic scores for fake samples, shape ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    torch.Tensor
        Scalar Wasserstein critic loss (minimised by the discriminator).
    """
    return y_pred.mean() - y_true.mean()


def gradient_penalty(
    discriminator: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP gradient penalty.

    Interpolates between ``real`` and ``fake`` samples and penalises the
    discriminator gradient norm for deviating from 1.

    Parameters
    ----------
    discriminator : nn.Module
        The discriminator / critic network.
    real : torch.Tensor
        Real samples, shape ``(N, C, D, H, W)``.
    fake : torch.Tensor
        Generated samples, same shape as ``real``.
    lambda_gp : float
        Penalty weight (default 10, standard WGAN-GP value).

    Returns
    -------
    torch.Tensor
        Scalar gradient penalty term.
    """
    b = real.size(0)
    eps = torch.rand(b, *([1] * (real.dim() - 1)), device=real.device)
    interp = (eps * real + (1.0 - eps) * fake.detach()).requires_grad_(True)
    d_interp = discriminator(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=list(range(1, real.dim()))) - 1) ** 2).mean()
    return lambda_gp * gp


# ---------------------------------------------------------------------------
# Class weights and weighted losses
# ---------------------------------------------------------------------------


def compute_class_weights(
    label_paths: list[str],
    n_classes: int,
    label_mapping: str | None = None,
    method: str = "inverse_frequency",
    max_samples: int | None = None,
) -> torch.Tensor:
    """Compute per-class weights from label volumes.

    Scans label files to count voxel frequencies per class, then converts
    to weights.  Useful for imbalanced segmentation (e.g., 50-class brain
    parcellation where small structures are underrepresented).

    Parameters
    ----------
    label_paths : list of str
        Paths to label NIfTI/MGZ files.
    n_classes : int
        Number of target classes.
    label_mapping : str or None
        Label mapping name (e.g., ``"50-class"``) or CSV path.
        If None, labels are used as-is.
    method : str
        ``"inverse_frequency"`` (1/freq, normalized) or
        ``"median_frequency"`` (median_freq/freq, as in SegNet).
    max_samples : int or None
        Limit scanning to this many files (for speed).

    Returns
    -------
    torch.Tensor
        Shape ``(n_classes,)`` float tensor of weights.
    """
    import nibabel as nib
    import numpy as np

    counts = np.zeros(n_classes, dtype=np.float64)
    paths = label_paths[:max_samples] if max_samples else label_paths

    remap_fn = None
    if label_mapping is not None:
        from nobrainer.processing.dataset import _load_label_mapping

        remap_fn = _load_label_mapping(label_mapping)

    for path in paths:
        arr = np.asarray(nib.load(path).dataobj, dtype=np.int32)
        if remap_fn is not None:
            arr = remap_fn(arr)
        for c in range(n_classes):
            counts[c] += (arr == c).sum()

    # Avoid division by zero
    counts = np.maximum(counts, 1.0)
    total = counts.sum()

    if method == "median_frequency":
        freqs = counts / total
        median_freq = np.median(freqs[freqs > 0])
        weights = median_freq / freqs
    else:
        # inverse_frequency: weight = total / (n_classes * count)
        weights = total / (n_classes * counts)

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def weighted_cross_entropy(
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.nn.CrossEntropyLoss:
    """Return a ``CrossEntropyLoss`` with optional per-class weights.

    Parameters
    ----------
    weight : torch.Tensor or None
        Per-class weights, shape ``(n_classes,)``.
    label_smoothing : float
        Label smoothing factor (default 0).
    """
    return torch.nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=label_smoothing,
    )


class HammingLoss(torch.nn.Module):
    """Hamming loss: fraction of misclassified voxels.

    A differentiable approximation of Hamming distance using soft
    predictions: ``g·(1-p) + (1-g)·p`` averaged over spatial dims.

    For use as a loss function with logits, set ``from_logits=True``
    to apply softmax first.

    Parameters
    ----------
    from_logits : bool
        Apply softmax to predictions (default True).
    """

    def __init__(self, from_logits: bool = True) -> None:
        super().__init__()
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.softmax(pred, dim=1)

        # One-hot encode target if needed
        if target.ndim == pred.ndim - 1:
            n_classes = pred.shape[1]
            target_oh = (
                torch.nn.functional.one_hot(target.long(), n_classes)
                .permute(0, 4, 1, 2, 3)
                .float()
            )
        else:
            target_oh = target.float()

        # Hamming: g*(1-p) + (1-g)*p = fraction of disagreement
        loss = target_oh * (1 - pred) + (1 - target_oh) * pred
        return loss.mean()


def hamming(from_logits: bool = True) -> HammingLoss:
    """Return a :class:`HammingLoss` instance."""
    return HammingLoss(from_logits=from_logits)


class DiceCELoss(torch.nn.Module):
    """Combined Dice + weighted CrossEntropy loss.

    Commonly used for imbalanced segmentation tasks.  The Dice component
    is inherently class-balanced; the CE component can use per-class
    weights.

    Parameters
    ----------
    weight : torch.Tensor or None
        Per-class weights for the CE term.
    dice_weight : float
        Relative weight of the Dice term (default 1.0).
    ce_weight : float
        Relative weight of the CE term (default 1.0).
    softmax : bool
        Apply softmax to predictions for the Dice term.
    label_smoothing : float
        Label smoothing for the CE term.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        softmax: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(softmax=softmax, to_onehot_y=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing
        )
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Dice expects target with channel dim
        if target.ndim == pred.ndim - 1:
            target_dice = target.unsqueeze(1)
        else:
            target_dice = target
        d = self.dice_loss(pred, target_dice)

        # CE expects target without channel dim, as long
        if target.ndim == pred.ndim:
            target_ce = target.squeeze(1)
        else:
            target_ce = target
        if target_ce.dtype != torch.long:
            target_ce = target_ce.long()
        ce = self.ce_loss(pred, target_ce)

        return self.dice_weight * d + self.ce_weight * ce


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
    "gradient_penalty": gradient_penalty,
    "hamming": hamming,
    "weighted_cross_entropy": weighted_cross_entropy,
    "dice_ce": DiceCELoss,
}


def get(name: str):
    """Return loss factory by name (case-insensitive)."""
    try:
        return _losses[name.lower()]
    except KeyError:
        avail = ", ".join(_losses)
        raise ValueError(f"Unknown loss '{name}'. Available: {avail}") from None
