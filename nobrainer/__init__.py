from . import (  # noqa: F401
    _version,
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

__version__ = _version.get_versions()["version"]
