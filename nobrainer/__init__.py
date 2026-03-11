from . import _version  # noqa: F401
from . import layers  # noqa: F401
from . import losses  # noqa: F401
from . import metrics  # noqa: F401

# The following modules are ported progressively (Phase 3 onwards)
# and will be added back as they are migrated to PyTorch:
#   dataset, io, models, prediction, training, transform, utils, volume

__version__ = _version.get_versions()["version"]
