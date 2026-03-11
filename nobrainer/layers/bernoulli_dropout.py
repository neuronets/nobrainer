"""Bernoulli dropout layer for PyTorch."""

import torch
import torch.nn as nn


class BernoulliDropout(nn.Module):
    """Bernoulli dropout layer.

    Multiplies input by a Bernoulli mask sampled with keep probability
    ``1 - rate``.  When ``scale_during_training`` is ``True`` the output
    is rescaled by ``1 / keep_prob`` so that the expected value is
    preserved (inverted dropout).  When it is ``False`` the raw Bernoulli
    mask is applied and the output is scaled by ``keep_prob`` at test
    time.

    Parameters
    ----------
    rate : float
        Drop probability (0 ≤ rate < 1).
    is_monte_carlo : bool
        When ``True`` the stochastic mask is applied regardless of
        ``training`` mode (enables MC-Dropout inference).
    scale_during_training : bool
        When ``True`` uses inverted dropout (scale at train time).
        When ``False`` scales at test time instead.
    seed : int or None
        Optional RNG seed (used to create a per-layer Generator).

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
        self.keep_prob = 1.0 - rate
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        apply_mask = self.is_monte_carlo or self.training
        if apply_mask:
            mask = torch.bernoulli(
                torch.full_like(x, self.keep_prob), generator=self._generator
            )
            out = x * mask
            return out / self.keep_prob if self.scale_during_training else out
        # deterministic path
        return x if self.scale_during_training else self.keep_prob * x

    def extra_repr(self) -> str:
        return (
            f"rate={self.rate}, is_monte_carlo={self.is_monte_carlo}, "
            f"scale_during_training={self.scale_during_training}"
        )
