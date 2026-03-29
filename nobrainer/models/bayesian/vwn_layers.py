"""Fully Factorized Gaussian (FFG) layers with local reparameterization.

These layers implement the convolution used in McClure et al. (2019),
Section 2.2.3.2 ("Spike-and-Slab Dropout with Learned Model Uncertainty"):

* Each weight has learnable mean ``μ_{f,t}`` and std ``σ_{f,t}``
* **Local reparameterization trick** (Kingma et al. 2015): instead of
  sampling weights, the output distribution is computed directly:
  ``output ~ N(conv(x, μ), conv(x², σ²))``  (Eqs. 12-14)
* The **spike-and-slab dropout (SSD)** model combines this with
  **concrete dropout** (Gal et al. 2017): ``output_v = b_f · (g_f * h)_v``
  where ``b_f`` is a per-filter concrete dropout mask (Eq. 11).

The KL divergence has two terms (Eq. 16 in paper):
  1. Bernoulli KL for concrete dropout gates (Eq. 17)
  2. Gaussian KL for each weight: ``KL(N(μ,σ) || N(μ_prior, σ_prior))`` (Eq. 18)

Prior parameters from the paper: ``p_prior=0.5, μ_prior=0, σ_prior=0.1``

Two dropout variants:
* **Bernoulli dropout** — standard ``nn.Dropout3d``, fixed rate (BD model)
* **Concrete dropout** — per-filter learnable drop rate (SSD model)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFGConv3d(nn.Module):
    """3-D convolution with Fully Factorized Gaussian (FFG) weights.

    Each weight has a learnable mean ``μ`` and std ``σ = |a|``.  During
    stochastic forward passes (``mc=True``), the **local reparameterization
    trick** (Kingma et al. 2015) computes the output distribution directly:

        ``μ* = conv(x, μ)``
        ``σ*² = conv(x², σ²)``
        ``output = μ* + σ* · ε,  ε ~ N(0, 1)``

    This matches Eqs. 12-14 of McClure et al. (2019).

    In deterministic mode (``mc=False``) only the mean path is used.

    Parameters
    ----------
    in_channels, out_channels : int
        Standard convolution channel counts.
    kernel_size : int
        Cubic kernel side length.
    stride, padding, dilation : int
        Standard ``nn.Conv3d`` arguments.
    bias : bool
        Whether to include a bias term (with its own sigma).
    sigma_init : float
        Initial value for ``|a|`` (default 1e-4, matching kwyk code).
    prior_mu : float
        Prior mean (default 0.0).
    prior_sigma : float
        Prior std (default 0.1, matching paper Section 2.2.3.2).
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
        sigma_init: float = 1e-4,
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        k = kernel_size
        weight_shape = (out_channels, in_channels, k, k, k)

        # Learnable mean μ_{f,t} per weight
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_normal_(self.weight_mu, mode="fan_in", nonlinearity="relu")

        # Learnable sigma: σ_{f,t} = |weight_a|
        # Initialized small so initial behavior is near-deterministic
        self.weight_a = nn.Parameter(torch.full(weight_shape, sigma_init))

        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_channels))
            self.bias_a = nn.Parameter(torch.full((out_channels,), sigma_init))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_a", None)

        # Accumulated KL (updated each forward pass)
        self.kl: torch.Tensor = torch.tensor(0.0)

    @property
    def weight_sigma(self) -> torch.Tensor:
        """Weight std: ``|weight_a|``."""
        return torch.abs(self.weight_a)

    def forward(self, x: torch.Tensor, mc: bool = True) -> torch.Tensor:
        """Forward pass with optional stochastic sampling.

        Parameters
        ----------
        x : Tensor
            Input tensor ``(B, C, D, H, W)``.
        mc : bool
            If True, use local reparameterization (stochastic).
            If False, use only the mean path (deterministic).
        """
        mu = self.weight_mu
        out_mean = F.conv3d(
            x, mu, self.bias_mu, self.stride, self.padding, self.dilation
        )

        if not mc:
            return out_mean

        # Local reparameterization trick (Eqs. 12-14 in McClure et al.)
        sigma = self.weight_sigma
        out_var = F.conv3d(
            x.pow(2), sigma.pow(2), None, self.stride, self.padding, self.dilation
        )
        if self.bias_a is not None:
            bias_sigma = torch.abs(self.bias_a)
            out_var = out_var + bias_sigma.pow(2).view(1, -1, 1, 1, 1)

        noise = torch.randn_like(out_mean)
        out = out_mean + torch.sqrt(out_var + 1e-8) * noise

        # KL(N(μ, σ) || N(μ_prior, σ_prior)) — Eq. 18
        self.kl = (
            torch.log(self.prior_sigma / (sigma + 1e-8))
            + (sigma.pow(2) + (mu - self.prior_mu).pow(2)) / (2 * self.prior_sigma**2)
            - 0.5
        ).sum()

        return out


# Backward-compatible alias
VWNConv3d = FFGConv3d


class ConcreteDropout3d(nn.Module):
    """Concrete dropout (Gal et al. 2017) with per-filter learnable rate.

    Instead of a fixed dropout probability, each output filter learns its
    own drop rate ``p`` via a continuous relaxation of Bernoulli sampling
    (Eq. 10 in McClure et al. 2019).

    Parameters
    ----------
    n_filters : int
        Number of filters (one ``p`` per filter).
    temperature : float
        Concrete distribution temperature (default 0.02, matching paper).
    init_p : float
        Initial dropout probability (default 0.9, matching kwyk code).
    prior_p : float
        Prior dropout probability for KL (default 0.5, matching paper).
    """

    def __init__(
        self,
        n_filters: int,
        temperature: float = 0.02,
        init_p: float = 0.9,
        prior_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.prior_p = prior_p
        # Store as raw logit; p = sigmoid(p_logit) to keep in (0, 1)
        init_logit = math.log(init_p / (1 - init_p + 1e-8))
        self.p_logit = nn.Parameter(torch.full((n_filters,), init_logit))

    @property
    def p(self) -> torch.Tensor:
        """Per-filter dropout probabilities, clamped to [0.05, 0.95]."""
        return torch.sigmoid(self.p_logit).clamp(0.05, 0.95)

    def forward(self, x: torch.Tensor, mc: bool = True) -> torch.Tensor:
        """Apply concrete dropout (Eq. 10).

        Parameters
        ----------
        x : Tensor
            Input ``(B, C, D, H, W)``.
        mc : bool
            If True, sample from concrete distribution.
            If False, scale by ``p`` (expectation).
        """
        p = self.p.view(1, -1, 1, 1, 1)

        if not mc:
            return x * p

        # Concrete relaxation of Bernoulli (Eq. 10)
        eps = 1e-8
        noise = torch.rand_like(x[:1])  # (1, C, D, H, W)
        z = torch.sigmoid(
            (
                torch.log(p + eps)
                - torch.log(1 - p + eps)
                + torch.log(noise + eps)
                - torch.log(1 - noise + eps)
            )
            / self.temperature
        )
        return x * z

    def kl_divergence(self) -> torch.Tensor:
        """KL(q_p || p_prior) for Bernoulli distributions (Eq. 17)."""
        p = self.p
        pp = self.prior_p
        eps = 1e-8
        return (
            p * torch.log(p / (pp + eps) + eps)
            + (1 - p) * torch.log((1 - p) / (1 - pp + eps) + eps)
        ).sum()

    def regularization(self) -> torch.Tensor:
        """Alias for kl_divergence (backward compat)."""
        return self.kl_divergence()
