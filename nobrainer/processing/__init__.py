"""Scikit-learn-style estimator API for nobrainer.

Provides high-level ``Segmentation``, ``Generation``, and ``Dataset``
classes that wrap the lower-level PyTorch internals.
"""

from .dataset import Dataset, extract_patches

__all__ = ["Dataset", "extract_patches"]

# Optional: Segmentation (requires core models)
try:
    from .segmentation import Segmentation  # noqa: F401

    __all__.append("Segmentation")
except ImportError:
    pass

# Optional: Generation (requires pytorch-lightning)
try:
    from .generation import Generation  # noqa: F401

    __all__.append("Generation")
except ImportError:
    pass
