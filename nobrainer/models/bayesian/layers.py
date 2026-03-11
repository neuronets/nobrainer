"""Bayesian convolutional and linear layers as Pyro modules.

Both ``BayesianConv3d`` and ``BayesianLinear`` implement weight
uncertainty by maintaining learnable ``weight_mu`` and ``weight_sigma``
parameters.  During each stochastic forward pass they sample a weight
matrix from ``Normal(weight_mu, softplus(weight_sigma))`` and accumulate
the KL divergence against the prior into ``self.kl``.
"""

from __future__ import annotations

import math

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
import torch
from torch.distributions import constraints
import torch.nn.functional as F


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
        ``"standard_normal"`` (σ=1) or ``"laplace"`` (approximated as
        tight Normal with σ=0.1 to keep KL analytic).
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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.prior_type = prior_type

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

        self.prior_sigma = 1.0 if prior_type == "standard_normal" else 0.1
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
        ``"standard_normal"`` or ``"laplace"``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_type: str = "standard_normal",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type

        std_init = math.sqrt(2.0 / in_features)
        self.weight_mu = PyroParam(
            torch.zeros(out_features, in_features).normal_(0, std_init),
            constraint=constraints.real,
        )
        self.weight_rho = PyroParam(
            torch.full((out_features, in_features), -3.0),
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

        self.prior_sigma = 1.0 if prior_type == "standard_normal" else 0.1
        self.kl: torch.Tensor = torch.tensor(0.0)

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = pyro.sample(
            f"{self._pyro_name}.weight",
            dist.Normal(self.weight_mu, self.weight_sigma + 1e-8).to_event(2),
        )
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
