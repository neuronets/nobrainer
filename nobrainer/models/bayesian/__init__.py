"""Bayesian model sub-package.

Two flavours of Bayesian convolution are provided:

* **Bayes-by-backprop** (``BayesianConv3d``, ``BayesianMeshNet``) — Pyro-based,
  weight uncertainty via learned mu/sigma, supports standard_normal/laplace priors.
* **Variational Weight Normalization** (``VWNConv3d``, ``KWYKMeshNet``) — matches
  the original kwyk architecture (McClure et al. 2019) with weight normalization,
  local reparameterization, and Bernoulli or Concrete dropout.
"""

from .bayesian_meshnet import BayesianMeshNet, bayesian_meshnet
from .bayesian_vnet import BayesianVNet, bayesian_vnet
from .kwyk_meshnet import KWYKMeshNet, kwyk_meshnet
from .layers import BayesianConv3d, BayesianLinear
from .utils import accumulate_kl
from .vwn_layers import ConcreteDropout3d, FFGConv3d, VWNConv3d

__all__ = [
    "BayesianConv3d",
    "BayesianLinear",
    "BayesianMeshNet",
    "BayesianVNet",
    "ConcreteDropout3d",
    "FFGConv3d",
    "KWYKMeshNet",
    "VWNConv3d",
    "accumulate_kl",
    "bayesian_meshnet",
    "bayesian_vnet",
    "kwyk_meshnet",
]
