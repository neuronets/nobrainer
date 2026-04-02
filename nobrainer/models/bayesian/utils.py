"""Utility functions for Bayesian models."""

from __future__ import annotations

import torch

from .layers import BayesianConv3d, BayesianLinear
from .vwn_layers import ConcreteDropout3d, FFGConv3d


def accumulate_kl(model: torch.nn.Module) -> torch.Tensor:
    """Sum KL divergence from all Bayesian layers in ``model``.

    Works with both Pyro-based models (BayesianConv3d, BayesianLinear)
    and VWN/FFG models (FFGConv3d, ConcreteDropout3d).

    Parameters
    ----------
    model : nn.Module
        A model containing one or more Bayesian layers.

    Returns
    -------
    torch.Tensor
        Scalar KL sum.
    """
    kl = torch.tensor(0.0)
    for m in model.modules():
        # Pyro-based layers
        if isinstance(m, (BayesianConv3d, BayesianLinear)):
            kl = kl + m.kl
        # VWN/FFG layers
        elif isinstance(m, FFGConv3d):
            kl = kl + m.kl
        # Concrete dropout regularization
        elif isinstance(m, ConcreteDropout3d):
            kl = kl + m.kl_divergence()
    return kl
