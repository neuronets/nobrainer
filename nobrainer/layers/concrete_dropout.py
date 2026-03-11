"""Concrete Dropout layer for PyTorch."""

import math

import torch
import torch.nn as nn


class ConcreteDropout(nn.Module):
    """Concrete (relaxed Bernoulli) dropout layer.

    Learns a per-channel drop probability ``p_post`` end-to-end via a
    differentiable relaxation of the Bernoulli mask.  A KL-divergence
    regulariser between ``p_post`` and a fixed prior ``p_prior = 0.5``
    is accumulated in ``self.kl_loss`` after each forward call.

    Parameters
    ----------
    in_channels : int
        Number of input channels (last dimension of the input tensor).
    is_monte_carlo : bool
        When ``True`` the stochastic concrete mask is applied regardless
        of ``training`` mode.
    temperature : float
        Temperature of the concrete distribution (lower → more binary).
    use_expectation : bool
        At test time, use ``x * p_post`` instead of the identity.
    scale_factor : float
        Normalisation factor for the KL regulariser.
    seed : int or None
        Optional RNG seed.

    References
    ----------
    Concrete Dropout. Y. Gal, J. Hron & A. Kendall, NeurIPS 2017.
    """

    def __init__(
        self,
        in_channels: int,
        is_monte_carlo: bool = False,
        temperature: float = 0.02,
        use_expectation: bool = False,
        scale_factor: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.is_monte_carlo = is_monte_carlo
        self.temperature = temperature
        self.use_expectation = use_expectation
        self.scale_factor = scale_factor
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

        # Learnable drop probability (per channel), initialised near 0.9
        self.p_logit = nn.Parameter(torch.full((in_channels,), math.log(0.9 / 0.1)))
        # Fixed prior p = 0.5 → logit = 0
        self.register_buffer("p_prior", torch.full((in_channels,), 0.5))
        self.kl_loss: torch.Tensor = torch.tensor(0.0)

    @property
    def p_post(self) -> torch.Tensor:
        """Dropout probability clipped to (0.05, 0.95)."""
        return torch.sigmoid(self.p_logit).clamp(0.05, 0.95)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        apply_mask = self.is_monte_carlo or self.training
        if apply_mask:
            out = self._apply_concrete(x)
        else:
            out = x * self.p_post if self.use_expectation else x
        self.kl_loss = self._kl_divergence()
        return out

    def _apply_concrete(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps
        p = self.p_post  # (C,)
        noise = torch.rand(
            x.shape, dtype=x.dtype, device=x.device, generator=self._generator
        ).clamp(eps, 1.0 - eps)
        z = torch.sigmoid(
            (
                torch.log(p + eps)
                - torch.log(1.0 - p + eps)
                + torch.log(noise)
                - torch.log(1.0 - noise)
            )
            / self.temperature
        )
        return x * z

    def _kl_divergence(self) -> torch.Tensor:
        eps = 1e-7
        p = self.p_post
        pr = self.p_prior
        kl = p * (torch.log(p + eps) - torch.log(pr + eps)) + (1 - p) * (
            torch.log(1 - p + eps) - torch.log(1 - pr + eps)
        )
        return kl.sum() / self.scale_factor

    def extra_repr(self) -> str:
        return (
            f"is_monte_carlo={self.is_monte_carlo}, temperature={self.temperature}, "
            f"scale_factor={self.scale_factor}"
        )
