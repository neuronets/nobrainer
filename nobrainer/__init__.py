try:
    from ._version import __version__  # noqa: F401
except (ImportError, ModuleNotFoundError):
    try:
        from . import _version  # noqa: F401

        __version__ = _version.get_versions()["version"]
    except (ImportError, AttributeError):
        __version__ = "0.0.0.dev0"

# Lazy imports: submodules are available via nobrainer.io, nobrainer.models, etc.
# but are not eagerly loaded to avoid requiring optional dependencies (monai,
# pyro-ppl, pytorch-lightning) at import time.
