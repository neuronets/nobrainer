"""KWYK MeshNet variants — matching McClure et al. (2019) architecture.

All three kwyk models use Fully Factorized Gaussian (FFG) convolutions
with learned per-weight μ and σ, and the local reparameterization trick
(Kingma et al. 2015).  They differ in the dropout layer:

* **bwn** / **bwn_multi**: FFG conv + Bernoulli dropout
  (``bwn`` disables dropout at inference; ``bwn_multi`` keeps it on)
* **bvwn_multi_prior**: FFG conv + Concrete dropout (learned per-filter rate)
  This is the "spike-and-slab dropout" (SSD) model from the paper.

Reference
---------
McClure P. et al., "Knowing What You Know in Brain Segmentation Using
Bayesian Deep Neural Networks", Front. Neuroinform. 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vwn_layers import ConcreteDropout3d, FFGConv3d

_DILATION_SCHEDULES: dict[int, list[int]] = {
    37: [1, 1, 1, 2, 4, 8, 1],
    67: [1, 1, 2, 4, 8, 16, 1],
    129: [1, 2, 4, 8, 16, 32, 1],
}


class _VWNLayerBernoulli(nn.Module):
    """VWN conv + ReLU + Bernoulli dropout (bwn / bwn_multi)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        dropout_rate: float,
        sigma_init: float,
    ) -> None:
        super().__init__()
        self.conv = FFGConv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
            sigma_init=sigma_init,
        )
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor, mc: bool = True) -> torch.Tensor:
        h = F.relu(self.conv(x, mc=mc))
        if mc:
            h = self.dropout(h)
        return h


class _VWNLayerConcrete(nn.Module):
    """VWN conv + ReLU + Concrete dropout (bvwn_multi_prior)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        sigma_init: float,
        concrete_temperature: float = 0.02,
        concrete_init_p: float = 0.9,
    ) -> None:
        super().__init__()
        self.conv = FFGConv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
            sigma_init=sigma_init,
        )
        self.dropout = ConcreteDropout3d(
            out_ch,
            temperature=concrete_temperature,
            init_p=concrete_init_p,
        )

    def forward(self, x: torch.Tensor, mc: bool = True) -> torch.Tensor:
        h = F.relu(self.conv(x, mc=mc))
        h = self.dropout(h, mc=mc)
        return h


class KWYKMeshNet(nn.Module):
    """KWYK MeshNet with variational weight normalization.

    This is the architecture used in McClure et al. (2019).  All layers
    use VWN convolutions; the ``dropout_type`` parameter selects between
    Bernoulli (``"bernoulli"``) and Concrete (``"concrete"``) dropout.

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input image channels.
    filters : int
        Feature-map count in all hidden layers.
    receptive_field : int
        One of ``37``, ``67``, ``129`` — selects the dilation schedule.
    dropout_type : str
        ``"bernoulli"`` for bwn/bwn_multi, ``"concrete"`` for bvwn_multi_prior.
    dropout_rate : float
        For Bernoulli dropout (ignored for concrete).
    sigma_init : float
        Initial value for weight sigma (default 1e-4, matching kwyk).
    concrete_temperature : float
        Temperature for concrete dropout (default 0.02).
    concrete_init_p : float
        Initial dropout probability for concrete dropout (default 0.9).
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        filters: int = 71,
        receptive_field: int = 67,
        dropout_type: str = "bernoulli",
        dropout_rate: float = 0.25,
        sigma_init: float = 1e-4,
        concrete_temperature: float = 0.02,
        concrete_init_p: float = 0.9,
    ) -> None:
        super().__init__()
        if receptive_field not in _DILATION_SCHEDULES:
            raise ValueError(
                f"receptive_field must be one of {list(_DILATION_SCHEDULES)}, "
                f"got {receptive_field}"
            )
        self.dropout_type = dropout_type
        dilations = _DILATION_SCHEDULES[receptive_field]
        self._n_layers = len(dilations)

        for i, dil in enumerate(dilations):
            in_ch = in_channels if i == 0 else filters
            if dropout_type == "concrete":
                layer = _VWNLayerConcrete(
                    in_ch,
                    filters,
                    dil,
                    sigma_init,
                    concrete_temperature,
                    concrete_init_p,
                )
            else:
                layer = _VWNLayerBernoulli(
                    in_ch,
                    filters,
                    dil,
                    dropout_rate,
                    sigma_init,
                )
            setattr(self, f"layer_{i}", layer)

        self.classifier = nn.Conv3d(filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, mc: bool = True) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input ``(B, 1, D, H, W)``.
        mc : bool
            If True, stochastic forward pass (variational + dropout).
            If False, deterministic (mean weights, no dropout).
        """
        h = x
        for i in range(self._n_layers):
            h = getattr(self, f"layer_{i}")(h, mc=mc)
        return self.classifier(h)

    def kl_divergence(self) -> torch.Tensor:
        """Sum KL divergence from all VWN conv layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, FFGConv3d):
                kl = kl + m.kl
        return kl

    def concrete_regularization(self) -> torch.Tensor:
        """Sum concrete dropout regularization (0 for bernoulli models)."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, ConcreteDropout3d):
                reg = reg + m.regularization()
        return reg


def kwyk_meshnet(
    n_classes: int = 1,
    in_channels: int = 1,
    filters: int = 71,
    receptive_field: int = 67,
    dropout_type: str = "bernoulli",
    dropout_rate: float = 0.25,
    sigma_init: float = 1e-4,
    concrete_temperature: float = 0.02,
    concrete_init_p: float = 0.9,
    **kwargs,
) -> KWYKMeshNet:
    """Factory function for :class:`KWYKMeshNet`."""
    return KWYKMeshNet(
        n_classes=n_classes,
        in_channels=in_channels,
        filters=filters,
        receptive_field=receptive_field,
        dropout_type=dropout_type,
        dropout_rate=dropout_rate,
        sigma_init=sigma_init,
        concrete_temperature=concrete_temperature,
        concrete_init_p=concrete_init_p,
    )


__all__ = ["KWYKMeshNet", "kwyk_meshnet"]
