from . import _version  # noqa: F401

__version__ = _version.get_versions()["version"]

# Lazy imports: submodules are available via nobrainer.io, nobrainer.models, etc.
# but are not eagerly loaded to avoid requiring optional dependencies (monai,
# pyro-ppl, pytorch-lightning) at import time.
