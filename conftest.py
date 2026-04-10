"""Root conftest.py — auto-skip GPU tests when CUDA is unavailable."""

from __future__ import annotations

import pytest
import torch


def _mps_supports_conv3d() -> bool:
    """Return True if Conv3D works on the MPS backend."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    try:
        m = torch.nn.Conv3d(1, 1, 1).to("mps")
        x = torch.randn(1, 1, 2, 2, 2, device="mps")
        m(x)
        return True
    except RuntimeError:
        return False


# Disable MPS auto-detection when Conv3D is unsupported (PyTorch < 2.3 on
# Apple Silicon).  Without this, ``nobrainer.gpu.get_device()`` returns MPS
# and every 3D convolution raises ``RuntimeError: Conv3D is not supported on
# MPS``.
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not _mps_supports_conv3d():
        torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with @pytest.mark.gpu when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available — skipping GPU test")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
