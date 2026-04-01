"""Bayesian MeshNet: dilated-convolution segmentation with weight uncertainty.

Replaces every ``nn.Conv3d`` in the 7-layer dilated architecture with
:class:`~nobrainer.models.bayesian.layers.BayesianConv3d`.

Reference
---------
Fedorov A. et al., "End-to-end learning of brain tissue segmentation
from imperfect labeling", IJCNN 2017. arXiv:1612.00940.
"""

from __future__ import annotations

from pyro.nn import PyroModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from nobrainer.models._constants import (  # noqa: E501
    DILATION_SCHEDULES as _DILATION_SCHEDULES,
)

from .layers import BayesianConv3d


class _BayesConvBNActDrop(PyroModule):
    """Single dilated Bayesian conv layer with BN + ELU/ReLU + spatial dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        activation: str,
        dropout_rate: float,
        prior_type: str,
        **sas_kwargs,
    ) -> None:
        super().__init__()
        padding = dilation  # same-size output for 3×3×3 kernel
        self.conv = BayesianConv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
            prior_type=prior_type,
            **sas_kwargs,
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.act_fn = {"relu": F.relu, "elu": F.elu}[activation.lower()]
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act_fn(self.bn(self.conv(x))))


class BayesianMeshNet(PyroModule):
    """3-D MeshNet with Bayesian convolutional layers.

    Identical dilated-convolution schedule as :class:`~nobrainer.models.meshnet.MeshNet`
    but all ``nn.Conv3d`` layers are replaced with :class:`BayesianConv3d`.

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
    activation : str
        ``"relu"`` or ``"elu"``.
    dropout_rate : float
        Spatial dropout probability (0 = disabled).
    prior_type : str
        ``"standard_normal"``, ``"laplace"``, or ``"spike_and_slab"``.
    kl_weight : float
        Scalar applied to the summed KL when computing the ELBO.
        Stored as an attribute; not used internally during forward.
    spike_sigma : float
        Spike component σ for spike-and-slab prior (default 0.001).
    slab_sigma : float
        Slab component σ for spike-and-slab prior (default 1.0).
    prior_pi : float
        Prior probability of the spike component (default 0.5).
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        filters: int = 71,
        receptive_field: int = 67,
        activation: str = "relu",
        dropout_rate: float = 0.25,
        prior_type: str = "standard_normal",
        kl_weight: float = 1.0,
        spike_sigma: float = 0.001,
        slab_sigma: float = 1.0,
        prior_pi: float = 0.5,
    ) -> None:
        super().__init__()
        if receptive_field not in _DILATION_SCHEDULES:
            raise ValueError(
                f"receptive_field must be one of {list(_DILATION_SCHEDULES)}, "
                f"got {receptive_field}"
            )
        self.kl_weight = kl_weight
        self.prior_type = prior_type
        dilations = _DILATION_SCHEDULES[receptive_field]
        self._n_layers = len(dilations)

        # Extra kwargs for spike-and-slab layers
        sas_kwargs = {}
        if prior_type == "spike_and_slab":
            sas_kwargs = {
                "spike_sigma": spike_sigma,
                "slab_sigma": slab_sigma,
                "prior_pi": prior_pi,
            }

        # Register each Bayesian layer as a named attribute so Pyro assigns
        # unique sample site names (nn.ModuleList does not propagate names).
        for i, dil in enumerate(dilations):
            in_ch = in_channels if i == 0 else filters
            layer = _BayesConvBNActDrop(
                in_ch,
                filters,
                dil,
                activation,
                dropout_rate,
                prior_type,
                **sas_kwargs,
            )
            setattr(self, f"layer_{i}", layer)

        # Final 1×1×1 classifier — deterministic
        self.classifier = nn.Conv3d(filters, n_classes, kernel_size=1)

    supports_mc = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i in range(self._n_layers):
            h = getattr(self, f"layer_{i}")(h)
        return self.classifier(h)


def bayesian_meshnet(
    n_classes: int = 1,
    in_channels: int = 1,
    filters: int = 71,
    receptive_field: int = 67,
    activation: str = "relu",
    dropout_rate: float = 0.25,
    prior_type: str = "standard_normal",
    kl_weight: float = 1.0,
    spike_sigma: float = 0.001,
    slab_sigma: float = 1.0,
    prior_pi: float = 0.5,
    **kwargs,
) -> BayesianMeshNet:
    """Factory function for :class:`BayesianMeshNet`."""
    return BayesianMeshNet(
        n_classes=n_classes,
        in_channels=in_channels,
        filters=filters,
        receptive_field=receptive_field,
        activation=activation,
        dropout_rate=dropout_rate,
        prior_type=prior_type,
        kl_weight=kl_weight,
        spike_sigma=spike_sigma,
        slab_sigma=slab_sigma,
        prior_pi=prior_pi,
    )


__all__ = ["BayesianMeshNet", "bayesian_meshnet"]
