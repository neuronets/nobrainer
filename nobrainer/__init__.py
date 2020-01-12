from distutils.version import LooseVersion

import tensorflow as tf

from nobrainer._version import get_versions
import nobrainer.dataset
import nobrainer.io
import nobrainer.layers
import nobrainer.losses
import nobrainer.metrics
import nobrainer.models
import nobrainer.prediction
import nobrainer.transform
import nobrainer.utils
import nobrainer.volume  # noqa

__version__ = get_versions()["version"]
del get_versions

if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    raise ValueError(
        "tensorflow>=2.0.0 must be installed but found version {}".format(
            tf.__version__
        )
    )
del LooseVersion, tf
