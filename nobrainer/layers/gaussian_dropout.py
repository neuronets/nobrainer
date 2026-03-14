"""Gaussian dropout layer for PyTorch."""

import math

import torch
import torch.nn as nn


class GaussianDropout(nn.Module):
    """Gaussian (multiplicative) dropout layer.

    Multiplies the input by noise sampled from ``Normal(1, σ²)`` where
    σ is derived from ``rate``.  When ``scale_during_training`` is
    ``True``, σ = sqrt(rate / (1 - rate)) (variance-preserving during
    training); otherwise σ = sqrt(rate * (1 - rate)).

    Parameters
    ----------
    rate : float
        Drop probability (0 ≤ rate < 1).
    is_monte_carlo : bool
        When ``True``, noise is applied regardless of ``training`` mode.
    scale_during_training : bool
        Selects which σ formula is used (see above).
    seed : int or None
        Optional RNG seed.

    References
    ----------
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    N. Srivastava et al., JMLR 2014.
    """

    def __init__(
        self,
        rate: float,
        is_monte_carlo: bool,
        scale_during_training: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"rate must be in [0, 1), got {rate}")
        self.rate = rate
        self.is_monte_carlo = is_monte_carlo
        self.scale_during_training = scale_during_training
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

        if scale_during_training:
            self._stddev = math.sqrt(rate / (1.0 - rate))
        else:
            self._stddev = math.sqrt(rate * (1.0 - rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_monte_carlo or self.training:
            noise = torch.randn_like(x, generator=self._generator) * self._stddev + 1.0
            return x * noise
        return x

    def extra_repr(self) -> str:
        return (
            f"rate={self.rate}, is_monte_carlo={self.is_monte_carlo}, "
            f"scale_during_training={self.scale_during_training}"
        )
