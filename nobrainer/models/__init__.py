"""Nobrainer model registry (PyTorch)."""

from pprint import pprint

from .autoencoder import autoencoder
from .bayesian import bayesian_meshnet, bayesian_vnet
from .generative import dcgan, progressivegan
from .highresnet import highresnet
from .meshnet import meshnet
from .segmentation import attention_unet, unet, unetr, vnet
from .simsiam import simsiam

__all__ = ["get", "list_available_models"]

_models = {
    "unet": unet,
    "vnet": vnet,
    "attention_unet": attention_unet,
    "unetr": unetr,
    "meshnet": meshnet,
    "highresnet": highresnet,
    "autoencoder": autoencoder,
    "simsiam": simsiam,
    "bayesian_vnet": bayesian_vnet,
    "bayesian_meshnet": bayesian_meshnet,
    "progressivegan": progressivegan,
    "dcgan": dcgan,
}


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
    try:
        return _models[name.lower()]
    except KeyError:
        avail = ", ".join(_models)
        raise ValueError(f"Unknown model '{name}'. Available: {avail}.") from None


def available_models() -> list[str]:
    return list(_models)


def list_available_models() -> None:
    pprint(available_models())
