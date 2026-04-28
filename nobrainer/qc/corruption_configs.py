"""Corruption configurations with severity levels.

Each corruption type has 5 severity levels with calibrated parameters.
Used by nobrainer.qc.corrupt for systematic QC evaluation.

K-space corruptions (motion, ghosting, spike) use TorchIO because
MONAI does not simulate k-space physics.

Image-space corruptions (noise, bias, blur, downsample, gamma) also
use TorchIO for consistency — the QC corruption pipeline operates on
tio.Subject objects, not MONAI dictionary transforms.

This does NOT duplicate nobrainer.augmentation, which uses MONAI
dictionary transforms for on-the-fly training augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torchio as tio


@dataclass(frozen=True)
class CorruptionConfig:
    """Configuration for a single corruption type with severity levels.

    Parameters:
        name: Human-readable corruption name.
        domain: "kspace" or "image".
        transform_class: TorchIO transform class.
        severity_params: Mapping from severity level (1-5) to transform kwargs.
    """

    name: str
    domain: str
    transform_class: type
    severity_params: dict[int, dict[str, Any]]

    def get_transform(self, severity: int) -> tio.Transform:
        """Create a TorchIO transform for the given severity level.

        Parameters:
            severity: Severity level (1-5).

        Returns:
            Configured TorchIO transform instance.
        """
        if severity not in self.severity_params:
            valid = sorted(self.severity_params.keys())
            raise ValueError(
                f"Severity {severity} not defined for '{self.name}'. Valid: {valid}"
            )
        return self.transform_class(**self.severity_params[severity])


def get_corruption_configs() -> dict[str, CorruptionConfig]:
    """Return all corruption configurations with 5 severity levels.

    Returns:
        Mapping from corruption name to its configuration.
    """
    return {
        # ── K-space corruptions (TorchIO only; MONAI has no equivalent) ──
        "motion": CorruptionConfig(
            name="motion",
            domain="kspace",
            transform_class=tio.RandomMotion,
            severity_params={
                1: {"degrees": 2, "translation": 1, "num_transforms": 2},
                2: {"degrees": 4, "translation": 3, "num_transforms": 4},
                3: {"degrees": 6, "translation": 5, "num_transforms": 6},
                4: {"degrees": 8, "translation": 7, "num_transforms": 8},
                5: {"degrees": 10, "translation": 10, "num_transforms": 10},
            },
        ),
        "ghosting": CorruptionConfig(
            name="ghosting",
            domain="kspace",
            transform_class=tio.RandomGhosting,
            severity_params={
                1: {"intensity": 0.2, "num_ghosts": 2},
                2: {"intensity": 0.4, "num_ghosts": 4},
                3: {"intensity": 0.6, "num_ghosts": 6},
                4: {"intensity": 0.8, "num_ghosts": 8},
                5: {"intensity": 1.0, "num_ghosts": 10},
            },
        ),
        "spike": CorruptionConfig(
            name="spike",
            domain="kspace",
            transform_class=tio.RandomSpike,
            severity_params={
                1: {"num_spikes": 1, "intensity": (0.5, 1.0)},
                2: {"num_spikes": 2, "intensity": (1.0, 2.0)},
                3: {"num_spikes": 3, "intensity": (2.0, 3.0)},
                4: {"num_spikes": 4, "intensity": (3.0, 4.0)},
                5: {"num_spikes": 5, "intensity": (4.0, 5.0)},
            },
        ),
        # ── Image-space corruptions (TorchIO for pipeline consistency) ──
        "noise": CorruptionConfig(
            name="noise",
            domain="image",
            transform_class=tio.RandomNoise,
            severity_params={
                1: {"std": (0.005, 0.01)},
                2: {"std": (0.01, 0.03)},
                3: {"std": (0.03, 0.06)},
                4: {"std": (0.06, 0.10)},
                5: {"std": (0.10, 0.15)},
            },
        ),
        "bias_field": CorruptionConfig(
            name="bias_field",
            domain="image",
            transform_class=tio.RandomBiasField,
            severity_params={
                1: {"coefficients": 0.1},
                2: {"coefficients": 0.3},
                3: {"coefficients": 0.5},
                4: {"coefficients": 0.7},
                5: {"coefficients": 1.0},
            },
        ),
        "blur": CorruptionConfig(
            name="blur",
            domain="image",
            transform_class=tio.RandomBlur,
            severity_params={
                1: {"std": (0.25, 0.5)},
                2: {"std": (0.5, 1.0)},
                3: {"std": (1.0, 2.0)},
                4: {"std": (2.0, 3.0)},
                5: {"std": (3.0, 4.0)},
            },
        ),
        "downsample": CorruptionConfig(
            name="downsample",
            domain="image",
            transform_class=tio.RandomAnisotropy,
            severity_params={
                1: {"downsampling": (1.5, 2.0)},
                2: {"downsampling": (2.0, 3.0)},
                3: {"downsampling": (3.0, 4.0)},
                4: {"downsampling": (4.0, 5.0)},
                5: {"downsampling": (5.0, 6.0)},
            },
        ),
        "gamma": CorruptionConfig(
            name="gamma",
            domain="image",
            transform_class=tio.RandomGamma,
            severity_params={
                1: {"log_gamma": (-0.1, 0.1)},
                2: {"log_gamma": (-0.2, 0.2)},
                3: {"log_gamma": (-0.3, 0.3)},
                4: {"log_gamma": (-0.5, 0.5)},
                5: {"log_gamma": (-0.7, 0.7)},
            },
        ),
    }
