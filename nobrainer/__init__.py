from . import _version

__version__ = _version.get_versions()["version"]

from . import (  # noqa: F401
    dataset,
    io,
    layers,
    losses,
    metrics,
    models,
    prediction,
    training,
    transform,
    utils,
    volume,
)
