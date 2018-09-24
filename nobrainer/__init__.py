# -*- coding: utf-8 -*-
"""Top-level module imports for nobrainer."""

import warnings
# Ignore FutureWarning (from h5py in this case).
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import tensorflow
except ImportError:
    raise ImportError(
        "TensorFlow cannot be found. Please re-install nobrainer with either"
        " the [cpu] or [gpu] extras, or install TensorFlow separately. Please"
        " see https://www.tensorflow.org/install/ for installation"
        " instructions.")

from nobrainer.io import read_csv
from nobrainer.io import read_json
from nobrainer.io import read_mapping
from nobrainer.io import read_volume
from nobrainer.io import save_json

from nobrainer.metrics import dice
from nobrainer.metrics import dice_numpy
from nobrainer.metrics import hamming
from nobrainer.metrics import hamming_numpy
from nobrainer.metrics import streaming_dice
from nobrainer.metrics import streaming_hamming

from nobrainer.models import get_estimator
from nobrainer.models import HighRes3DNet
from nobrainer.models import MeshNet
from nobrainer.models import QuickNAT
from nobrainer.models import UNet3D

from nobrainer.predict import predict

from nobrainer.train import train

from nobrainer.volume import binarize
from nobrainer.volume import change_brightness
from nobrainer.volume import downsample
from nobrainer.volume import flip
from nobrainer.volume import from_blocks
from nobrainer.volume import iterblocks_3d
from nobrainer.volume import itervolumes
from nobrainer.volume import normalize_zero_one
from nobrainer.volume import reduce_contrast
from nobrainer.volume import replace
from nobrainer.volume import rotate
from nobrainer.volume import salt_and_pepper
from nobrainer.volume import shift
from nobrainer.volume import to_blocks
from nobrainer.volume import zoom
from nobrainer.volume import zscore
from nobrainer.volume import VolumeDataGenerator
