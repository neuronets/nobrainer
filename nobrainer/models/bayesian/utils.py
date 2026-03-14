"""Utility functions for Bayesian models."""

from __future__ import annotations

import torch

from .layers import BayesianConv3d, BayesianLinear


def accumulate_kl(model: torch.nn.Module) -> torch.Tensor:
    """Sum KL divergence from all Bayesian layers in ``model``.

    Iterates all named sub-modules and adds the ``.kl`` attribute from
    any :class:`BayesianConv3d` or :class:`BayesianLinear` layers.
    These attributes are populated during the forward pass.

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
        if isinstance(m, (BayesianConv3d, BayesianLinear)):
            kl = kl + m.kl
    return kl
