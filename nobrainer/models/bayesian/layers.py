"""Bayesian convolutional and linear layers as Pyro modules.

Both ``BayesianConv3d`` and ``BayesianLinear`` implement weight
uncertainty by maintaining learnable ``weight_mu`` and ``weight_sigma``
parameters.  During each stochastic forward pass they sample a weight
matrix from ``Normal(weight_mu, softplus(weight_sigma))`` and accumulate
the KL divergence against the prior into ``self.kl``.

Three prior types are supported (matching the kwyk study variants):

* ``"standard_normal"`` — N(0, 1) prior, standard Bayes-by-backprop.
* ``"laplace"`` — tight Normal N(0, 0.1) approximation of a Laplace prior.
* ``"spike_and_slab"`` — mixture prior ``π·N(0, σ₁) + (1-π)·N(0, σ₂)``
  where σ₁ (spike) is small and σ₂ (slab) is large.  Each weight also
  learns a log-odds ``z_logit`` controlling how much mass is on the spike
  vs slab, implementing variational spike-and-slab dropout (SSD) as in
  McClure et al. (2019).
"""

from __future__ import annotations

import math

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
import torch
from torch.distributions import constraints
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# KL helpers
# ---------------------------------------------------------------------------


def _kl_normal_normal(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    prior_mu: float,
    prior_sigma: float,
) -> torch.Tensor:
    """Analytic KL(N(mu, sigma) || N(prior_mu, prior_sigma))."""
    return (
        torch.log(prior_sigma / (sigma + 1e-8))
        + (sigma**2 + (mu - prior_mu) ** 2) / (2 * prior_sigma**2)
        - 0.5
    ).sum()


def _kl_spike_and_slab(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    z_logit: torch.Tensor,
    spike_sigma: float,
    slab_sigma: float,
    prior_pi: float,
) -> torch.Tensor:
    """KL divergence for spike-and-slab variational posterior.

    The variational posterior is:
        q(w, z) = Bernoulli(z; sigmoid(z_logit)) · N(w; mu, sigma)

    The prior is:
        p(w, z) = (pi·N(0, spike_sigma) + (1-pi)·N(0, slab_sigma))

    We use the closed-form approximation from Louizos et al. (2017) and
    the practical version used in the kwyk spike-and-slab dropout.
    """
    z = torch.sigmoid(z_logit)

    # Log-likelihood under spike and slab components
    log_spike = -0.5 * math.log(2 * math.pi * spike_sigma**2) - (
        mu**2 + sigma**2
    ) / (2 * spike_sigma**2)
    log_slab = -0.5 * math.log(2 * math.pi * slab_sigma**2) - (
        mu**2 + sigma**2
    ) / (2 * slab_sigma**2)

    # Entropy of the Bernoulli gate
    entropy_z = -(z * torch.log(z + 1e-8) + (1 - z) * torch.log(1 - z + 1e-8))

    # KL = E_q[log q - log p]
    # log q(w|z=slab) - log p(w) where p is the mixture
    kl_per_weight = (
        z * (-0.5 * torch.log(2 * math.pi * sigma**2 + 1e-8) - 0.5 - log_slab)
        + (1 - z) * (-log_spike)
        - entropy_z
        + z * math.log(1 - prior_pi + 1e-8)
        + (1 - z) * math.log(prior_pi + 1e-8)
    )

    return kl_per_weight.sum()


# ---------------------------------------------------------------------------
# Bayesian layers
# ---------------------------------------------------------------------------


class BayesianConv3d(PyroModule):
    """3-D convolution with learnable weight distribution (Pyro).

    Parameters
    ----------
    in_channels, out_channels : int
        Standard convolution channel counts.
    kernel_size : int
        Cubic kernel side length.
    stride, padding, dilation : int
        Standard ``nn.Conv3d`` arguments.
    bias : bool
        Whether to include a deterministic bias term.
    prior_type : str
        ``"standard_normal"`` (σ=1), ``"laplace"`` (tight Normal σ=0.1),
        or ``"spike_and_slab"`` (mixture prior with learnable gates).
    spike_sigma : float
        Spike component σ for spike-and-slab prior (default 0.001).
    slab_sigma : float
        Slab component σ for spike-and-slab prior (default 1.0).
    prior_pi : float
        Prior probability of the spike component (default 0.5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        prior_type: str = "standard_normal",
        spike_sigma: float = 0.001,
        slab_sigma: float = 1.0,
        prior_pi: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.prior_type = prior_type
        self.spike_sigma = spike_sigma
        self.slab_sigma = slab_sigma
        self.prior_pi = prior_pi

        weight_shape = (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        )
        # Kaiming init for mu
        fan_in = in_channels * kernel_size**3
        std_init = math.sqrt(2.0 / fan_in)
        self.weight_mu = PyroParam(
            torch.zeros(weight_shape).normal_(0, std_init),
            constraint=constraints.real,
        )
        self.weight_rho = PyroParam(
            torch.full(weight_shape, -3.0),  # softplus(-3) ≈ 0.05
            constraint=constraints.real,
        )

        # Spike-and-slab gate logits (one per weight)
        if prior_type == "spike_and_slab":
            self.z_logit = PyroParam(
                torch.full(weight_shape, 2.0),  # sigmoid(2) ≈ 0.88 → mostly slab
                constraint=constraints.real,
            )

        if bias:
            self.bias_mu = PyroParam(
                torch.zeros(out_channels), constraint=constraints.real
            )
            self.bias_rho = PyroParam(
                torch.full((out_channels,), -3.0), constraint=constraints.real
            )
        else:
            self.bias_mu = None
            self.bias_rho = None

        if prior_type == "standard_normal":
            self.prior_sigma = 1.0
        elif prior_type == "laplace":
            self.prior_sigma = 0.1
        else:
            self.prior_sigma = slab_sigma  # used as fallback only
        self.kl: torch.Tensor = torch.tensor(0.0)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = pyro.sample(
            f"{self._pyro_name}.weight",
            dist.Normal(self.weight_mu, self.weight_sigma + 1e-8).to_event(
                self.weight_mu.dim()
            ),
        )

        if self.prior_type == "spike_and_slab":
            # Apply spike-and-slab mask: sample Bernoulli gate, mask weights
            z_prob = torch.sigmoid(self.z_logit)
            z_mask = torch.bernoulli(z_prob)
            weight = weight * z_mask
            self.kl = _kl_spike_and_slab(
                self.weight_mu,
                self.weight_sigma,
                self.z_logit,
                self.spike_sigma,
                self.slab_sigma,
                self.prior_pi,
            )
        else:
            self.kl = _kl_normal_normal(
                self.weight_mu, self.weight_sigma, 0.0, self.prior_sigma
            )

        bias = None
        if self.bias_mu is not None:
            bias_sigma = F.softplus(self.bias_rho)
            bias = pyro.sample(
                f"{self._pyro_name}.bias",
                dist.Normal(self.bias_mu, bias_sigma + 1e-8).to_event(1),
            )
            self.kl = self.kl + _kl_normal_normal(
                self.bias_mu, bias_sigma, 0.0, self.prior_sigma
            )

        return F.conv3d(x, weight, bias, self.stride, self.padding, self.dilation)


class BayesianLinear(PyroModule):
    """Fully-connected layer with learnable weight distribution (Pyro).

    Parameters
    ----------
    in_features, out_features : int
        Standard ``nn.Linear`` dimensions.
    bias : bool
        Whether to include a deterministic bias term.
    prior_type : str
        ``"standard_normal"``, ``"laplace"``, or ``"spike_and_slab"``.
    spike_sigma : float
        Spike component σ for spike-and-slab prior (default 0.001).
    slab_sigma : float
        Slab component σ for spike-and-slab prior (default 1.0).
    prior_pi : float
        Prior probability of the spike component (default 0.5).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_type: str = "standard_normal",
        spike_sigma: float = 0.001,
        slab_sigma: float = 1.0,
        prior_pi: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type
        self.spike_sigma = spike_sigma
        self.slab_sigma = slab_sigma
        self.prior_pi = prior_pi

        std_init = math.sqrt(2.0 / in_features)
        self.weight_mu = PyroParam(
            torch.zeros(out_features, in_features).normal_(0, std_init),
            constraint=constraints.real,
        )
        self.weight_rho = PyroParam(
            torch.full((out_features, in_features), -3.0),
            constraint=constraints.real,
        )

        if prior_type == "spike_and_slab":
            self.z_logit = PyroParam(
                torch.full((out_features, in_features), 2.0),
                constraint=constraints.real,
            )

        if bias:
            self.bias_mu = PyroParam(
                torch.zeros(out_features), constraint=constraints.real
            )
            self.bias_rho = PyroParam(
                torch.full((out_features,), -3.0), constraint=constraints.real
            )
        else:
            self.bias_mu = None
            self.bias_rho = None

        if prior_type == "standard_normal":
            self.prior_sigma = 1.0
        elif prior_type == "laplace":
            self.prior_sigma = 0.1
        else:
            self.prior_sigma = slab_sigma
        self.kl: torch.Tensor = torch.tensor(0.0)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = pyro.sample(
            f"{self._pyro_name}.weight",
            dist.Normal(self.weight_mu, self.weight_sigma + 1e-8).to_event(2),
        )

        if self.prior_type == "spike_and_slab":
            z_prob = torch.sigmoid(self.z_logit)
            z_mask = torch.bernoulli(z_prob)
            weight = weight * z_mask
            self.kl = _kl_spike_and_slab(
                self.weight_mu,
                self.weight_sigma,
                self.z_logit,
                self.spike_sigma,
                self.slab_sigma,
                self.prior_pi,
            )
        else:
            self.kl = _kl_normal_normal(
                self.weight_mu, self.weight_sigma, 0.0, self.prior_sigma
            )

        bias = None
        if self.bias_mu is not None:
            bias_sigma = F.softplus(self.bias_rho)
            bias = pyro.sample(
                f"{self._pyro_name}.bias",
                dist.Normal(self.bias_mu, bias_sigma + 1e-8).to_event(1),
            )
            self.kl = self.kl + _kl_normal_normal(
                self.bias_mu, bias_sigma, 0.0, self.prior_sigma
            )

        return F.linear(x, weight, bias)
