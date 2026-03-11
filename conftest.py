"""Root conftest.py — auto-skip GPU tests when CUDA is unavailable."""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with @pytest.mark.gpu when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available — skipping GPU test")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip_gpu)
