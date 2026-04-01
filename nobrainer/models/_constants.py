"""Shared constants for nobrainer models."""

from __future__ import annotations

# Dilation schedules indexed by receptive field size.
# Used by MeshNet, BayesianMeshNet, and KWYKMeshNet.
DILATION_SCHEDULES: dict[int, list[int]] = {
    37: [1, 1, 1, 2, 4, 8, 1],
    67: [1, 1, 2, 4, 8, 16, 1],
    129: [1, 2, 4, 8, 16, 32, 1],
}
