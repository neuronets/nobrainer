"""Nobrainer model registry (PyTorch)."""

from pprint import pprint

from .autoencoder import autoencoder
from .highresnet import highresnet
from .meshnet import meshnet
from .segformer3d import segformer3d
from .segmentation import attention_unet, segresnet, swin_unetr, unet, unetr, vnet
from .simsiam import simsiam

__all__ = ["get", "list_available_models"]

# Core models (always available)
_models = {
    "unet": unet,
    "vnet": vnet,
    "attention_unet": attention_unet,
    "unetr": unetr,
    "meshnet": meshnet,
    "highresnet": highresnet,
    "autoencoder": autoencoder,
    "simsiam": simsiam,
    "swin_unetr": swin_unetr,
    "segresnet": segresnet,
    "segformer3d": segformer3d,
}

# Optional: Bayesian models (require pyro-ppl)
try:
    from .bayesian import bayesian_meshnet, bayesian_vnet

    _models["bayesian_vnet"] = bayesian_vnet
    _models["bayesian_meshnet"] = bayesian_meshnet
except ImportError:
    pass

# KWYK MeshNet (VWN-based, no Pyro dependency)
from .bayesian.kwyk_meshnet import kwyk_meshnet  # noqa: E402

_models["kwyk_meshnet"] = kwyk_meshnet

# Optional: Generative models (require pytorch-lightning)
try:
    from .generative import dcgan, progressivegan

    _models["progressivegan"] = progressivegan
    _models["dcgan"] = dcgan
except ImportError:
    pass


def get(name: str):
    """Return factory callable for a model by name (case-insensitive).

    Parameters
    ----------
    name : str
        Model name.

    Returns
    -------
    Callable that constructs a ``torch.nn.Module``.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    key = name.lower()
    if key in _models:
        return _models[key]
    # Check if it's an optional model that wasn't loaded
    optional = {
        "bayesian_vnet": "pyro-ppl",
        "bayesian_meshnet": "pyro-ppl",
        "progressivegan": "pytorch-lightning",
        "dcgan": "pytorch-lightning",
    }
    if key in optional:
        raise ImportError(
            f"Model '{name}' requires '{optional[key]}'. "
            f"Install with: uv pip install {optional[key]}"
        )
    avail = ", ".join(_models)
    raise ValueError(f"Unknown model '{name}'. Available: {avail}.")


def available_models() -> list[str]:
    return list(_models)


def list_available_models() -> None:
    pprint(available_models())
