from pprint import pprint

from .attention_unet import attention_unet
from .attention_unet_with_inception import attention_unet_with_inception
from .autoencoder import autoencoder
from .bayesian_meshnet import variational_meshnet
from .dcgan import dcgan
from .highresnet import highresnet
from .meshnet import meshnet
from .progressiveae import progressiveae
from .progressivegan import progressivegan
from .unet import unet
from .unetr import unetr

__all__ = ["get", "list_available_models"]

_models = {
    "highresnet": highresnet,
    "meshnet": meshnet,
    "unet": unet,
    "autoencoder": autoencoder,
    "progressivegan": progressivegan,
    "progressiveae": progressiveae,
    "dcgan": dcgan,
    "attention_unet": attention_unet,
    "attention_unet_with_inception": attention_unet_with_inception,
    "unetr": unetr,
    "variational_meshnet": variational_meshnet,
}


def get(name):
    """Return callable that creates a particular `tf.keras.Model`.

    Parameters
    ----------
    name: str, the name of the model (case-insensitive).

    Returns
    -------
    Callable, which instantiates a `tf.keras.Model` object.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")

    try:
        return _models[name.lower()]
    except KeyError:
        avail = ", ".join(_models.keys())
        raise ValueError(
            "Unknown model: '{}'. Available models are {}.".format(name, avail)
        )


def available_models():
    return list(_models)


def list_available_models():
    pprint(available_models())
