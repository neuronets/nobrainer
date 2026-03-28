"""Warm-start a Bayesian model from a trained deterministic model."""

from __future__ import annotations

import logging

import torch.nn as nn

from nobrainer.models.bayesian.layers import BayesianConv3d

logger = logging.getLogger(__name__)


def warmstart_bayesian_from_deterministic(
    bayesian_model: nn.Module,
    deterministic_model: nn.Module,
    initial_rho: float = -3.0,
) -> int:
    """Transfer deterministic Conv3d weights to BayesianConv3d weight_mu.

    Matches layers by position (not name) since the deterministic MeshNet
    uses ``nn.Sequential`` (``encoder.N.block.0``) while the Bayesian
    MeshNet uses named attributes (``layer_N.conv``).

    For each matching pair:

    * **Conv3d -> BayesianConv3d**: copies ``weight`` to ``weight_mu``,
      fills ``weight_rho`` with *initial_rho*, and handles bias if
      present.
    * **BatchNorm3d -> BatchNorm3d**: copies ``weight``, ``bias``,
      ``running_mean``, and ``running_var``.

    Parameters
    ----------
    bayesian_model : nn.Module
        Target Bayesian model whose parameters will be overwritten.
    deterministic_model : nn.Module
        Source deterministic model with trained weights.
    initial_rho : float, optional
        Value to fill ``weight_rho`` (and ``bias_rho``) with.
        ``softplus(-3.0) ≈ 0.05``.  Default is ``-3.0``.

    Returns
    -------
    int
        Number of layers whose weights were transferred.
    """
    # First try name-based matching (works if architectures share naming)
    transferred = _transfer_by_name(bayesian_model, deterministic_model, initial_rho)
    if transferred > 0:
        return transferred

    # Fall back to positional matching (different naming conventions)
    return _transfer_by_position(bayesian_model, deterministic_model, initial_rho)


def _transfer_by_name(
    bayesian_model: nn.Module,
    deterministic_model: nn.Module,
    initial_rho: float,
) -> int:
    """Match layers by module name."""
    det_modules = dict(deterministic_model.named_modules())
    bayes_modules = dict(bayesian_model.named_modules())
    transferred = 0

    for name, bayes_mod in bayes_modules.items():
        if name not in det_modules:
            continue
        det_mod = det_modules[name]
        transferred += _transfer_pair(det_mod, bayes_mod, name, initial_rho)

    if transferred > 0:
        logger.info(
            "Warm-started %d layers (name-matched) from deterministic model.",
            transferred,
        )
    return transferred


def _transfer_by_position(
    bayesian_model: nn.Module,
    deterministic_model: nn.Module,
    initial_rho: float,
) -> int:
    """Match Conv3d/BN layers by position (order of appearance)."""
    # Collect Conv3d layers from deterministic model
    det_convs = [
        (n, m)
        for n, m in deterministic_model.named_modules()
        if isinstance(m, nn.Conv3d)
    ]
    det_bns = [
        (n, m)
        for n, m in deterministic_model.named_modules()
        if isinstance(m, nn.BatchNorm3d)
    ]

    # Collect BayesianConv3d layers from Bayesian model
    bayes_convs = [
        (n, m)
        for n, m in bayesian_model.named_modules()
        if isinstance(m, BayesianConv3d)
    ]
    bayes_bns = [
        (n, m)
        for n, m in bayesian_model.named_modules()
        if isinstance(m, nn.BatchNorm3d)
    ]

    transferred = 0

    # Transfer Conv3d -> BayesianConv3d by position
    for i, ((det_name, det_conv), (bay_name, bay_conv)) in enumerate(
        zip(det_convs, bayes_convs)
    ):
        if det_conv.weight.shape != bay_conv.weight_mu.shape:
            logger.warning(
                "Shape mismatch at position %d: det %s %s vs bay %s %s",
                i,
                det_name,
                det_conv.weight.shape,
                bay_name,
                bay_conv.weight_mu.shape,
            )
            continue

        bay_conv.weight_mu.data.copy_(det_conv.weight.data)
        bay_conv.weight_rho.data.fill_(initial_rho)

        if det_conv.bias is not None and bay_conv.bias_mu is not None:
            bay_conv.bias_mu.data.copy_(det_conv.bias.data)
            bay_conv.bias_rho.data.fill_(initial_rho)

        transferred += 1
        logger.debug("Transferred Conv3d[%d] %s -> %s", i, det_name, bay_name)

    # Transfer BatchNorm3d by position
    for i, ((det_name, det_bn), (bay_name, bay_bn)) in enumerate(
        zip(det_bns, bayes_bns)
    ):
        if det_bn.weight is not None and bay_bn.weight is not None:
            bay_bn.weight.data.copy_(det_bn.weight.data)
        if det_bn.bias is not None and bay_bn.bias is not None:
            bay_bn.bias.data.copy_(det_bn.bias.data)
        if det_bn.running_mean is not None:
            bay_bn.running_mean.copy_(det_bn.running_mean)
        if det_bn.running_var is not None:
            bay_bn.running_var.copy_(det_bn.running_var)

        transferred += 1
        logger.debug("Transferred BatchNorm3d[%d] %s -> %s", i, det_name, bay_name)

    logger.info(
        "Warm-started %d layers (position-matched) from deterministic model.",
        transferred,
    )
    return transferred


def _transfer_pair(
    det_mod: nn.Module,
    bayes_mod: nn.Module,
    name: str,
    initial_rho: float,
) -> int:
    """Transfer weights for a single matching pair. Returns 1 if transferred."""
    is_conv = isinstance(det_mod, nn.Conv3d)
    is_bayes_conv = isinstance(bayes_mod, BayesianConv3d)

    if is_conv and is_bayes_conv:
        bayes_mod.weight_mu.data.copy_(det_mod.weight.data)
        bayes_mod.weight_rho.data.fill_(initial_rho)

        if det_mod.bias is not None and bayes_mod.bias_mu is not None:
            bayes_mod.bias_mu.data.copy_(det_mod.bias.data)
            bayes_mod.bias_rho.data.fill_(initial_rho)

        logger.debug("Transferred Conv3d weights: %s", name)
        return 1

    if isinstance(det_mod, nn.BatchNorm3d) and isinstance(bayes_mod, nn.BatchNorm3d):
        if det_mod.weight is not None and bayes_mod.weight is not None:
            bayes_mod.weight.data.copy_(det_mod.weight.data)
        if det_mod.bias is not None and bayes_mod.bias is not None:
            bayes_mod.bias.data.copy_(det_mod.bias.data)
        if det_mod.running_mean is not None:
            bayes_mod.running_mean.copy_(det_mod.running_mean)
        if det_mod.running_var is not None:
            bayes_mod.running_var.copy_(det_mod.running_var)

        logger.debug("Transferred BatchNorm3d params: %s", name)
        return 1

    return 0
