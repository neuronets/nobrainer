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

    Iterates both models' ``named_modules()`` in parallel and, for each
    matching pair:

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
    det_modules = dict(deterministic_model.named_modules())
    bayes_modules = dict(bayesian_model.named_modules())

    transferred = 0

    for name, bayes_mod in bayes_modules.items():
        if name not in det_modules:
            continue
        det_mod = det_modules[name]

        # Conv3d -> BayesianConv3d
        is_conv = isinstance(det_mod, nn.Conv3d)
        is_bayes_conv = isinstance(bayes_mod, BayesianConv3d)
        if is_conv and is_bayes_conv:
            bayes_mod.weight_mu.data.copy_(det_mod.weight.data)
            bayes_mod.weight_rho.data.fill_(initial_rho)

            if det_mod.bias is not None and bayes_mod.bias_mu is not None:
                bayes_mod.bias_mu.data.copy_(det_mod.bias.data)
                bayes_mod.bias_rho.data.fill_(initial_rho)

            transferred += 1
            logger.debug("Transferred Conv3d weights: %s", name)

        # BatchNorm3d -> BatchNorm3d
        elif isinstance(det_mod, nn.BatchNorm3d) and isinstance(
            bayes_mod, nn.BatchNorm3d
        ):
            if det_mod.weight is not None and bayes_mod.weight is not None:
                bayes_mod.weight.data.copy_(det_mod.weight.data)
            if det_mod.bias is not None and bayes_mod.bias is not None:
                bayes_mod.bias.data.copy_(det_mod.bias.data)
            if det_mod.running_mean is not None:
                bayes_mod.running_mean.copy_(det_mod.running_mean)
            if det_mod.running_var is not None:
                bayes_mod.running_var.copy_(det_mod.running_var)

            transferred += 1
            logger.debug("Transferred BatchNorm3d params: %s", name)

    logger.info(
        "Warm-started %d layers from deterministic model.",
        transferred,
    )
    return transferred
