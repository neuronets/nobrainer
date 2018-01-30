import os
from warnings import warn

import nibabel as nib
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K

from highres3dnet import dice_loss, HighRes3DNet

# Configuration
# -------------
NUM_CLASSES = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WINDOW_SHAPE = (128, 128, 128)
NUM_CHANNELS = 1
INPUT_SHAPE = (*WINDOW_SHAPE, NUM_CHANNELS)
TARGET_DTYPE = 'uint8'
CSV_FILEPATH = (
    "/om2/user/jakubk/openmind-surface-data/file-lists/master_file_list_brainmask.csv"
)
TENSORBOARD_BASE_DIR = (
    "/om/user/jakubk/nobrainer-code/niftynet_to_keras/models"
)


sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_last')


def _get_timestamp():
    import datetime
    return str(datetime.datetime.now()).split('.')[0].replace(' ', '_')


def get_tensorboard_dir(base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()
    window = "_".join(str(ii) for ii in WINDOW_SHAPE)
    rel_dir = (
        "highres3dnet-{num_classes}_classes-{lr}_lr-{batch}_batch-"
        "{window}_window-{ts}"
    ).format(
        num_classes=NUM_CLASSES, lr=LEARNING_RATE, batch=BATCH_SIZE,
        window=window, ts=_get_timestamp())
    return os.path.join(base_dir, rel_dir, 'logs')


def load_volume(filepath, return_affine=False, c_contiguous=True, dtype=None):
    """Return data given filepath to volume. Optionally return affine array.

    Making the data array contiguous takes more time during loading, but this
    ultimately saves time when viewing blocks of data with `skimage`.
    """
    img = nib.load(filepath)
    data = np.asarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    img.uncache()
    if c_contiguous:
        data = np.ascontiguousarray(data)
    if return_affine:
        return data, img.affine
    return data


def one_hot(a, **kwargs):
    """Return one-hot array of N-D array `a`."""
    # https://stackoverflow.com/a/37323404/5666087
    n_values = np.max(a) + 1
    return np.eye(n_values, **kwargs)[a]


def _preprocess_data(data):
    data = view_as_blocks(data, WINDOW_SHAPE).reshape(-1, *WINDOW_SHAPE)
    return data[Ellipsis, np.newaxis]


def _preprocess_target(target):
    target = one_hot(target, dtype=TARGET_DTYPE)
    new_shape = (*WINDOW_SHAPE, NUM_CLASSES)
    return view_as_blocks(target, new_shape).reshape(-1, *new_shape)


def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).
    Blocks are non-overlapping views of the input array.

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    block_shape : tuple

    Notes
    -----
    Copied from `skimage.util.view_as_blocks` to avoid having to install the
    entire package + dependencies.
    """
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view

    if not arr_in.flags.contiguous:
        warn(RuntimeWarning("Cannot provide views on a non-contiguous input "
                            "array without copying."))

    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


df_input = pd.read_csv(CSV_FILEPATH)

model = HighRes3DNet(n_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)

# Use multiple GPUs.
# gpu_ids = [int(ss) for ss in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
# model = keras.utils.multi_gpu_model(model, gpus=gpu_ids)

adam = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(adam, dice_loss)


# https://github.com/keras-team/keras/issues/5935#issuecomment-289041967
class MemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        import resource
        # max resident set size
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage = usage * resource.getpagesize() / 1000000.0
        print("Usage: {:0.0f} Mb".format(usage))


_tensorboard_dir = get_tensorboard_dir(base_dir=TENSORBOARD_BASE_DIR)

print("++ Saving Tensorboard information to\n{}".format(_tensorboard_dir))

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=_tensorboard_dir,
        write_graph=False,
        batch_size=BATCH_SIZE,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(_tensorboard_dir, "..", "model-{epoch:02d}.h5"),
        period=50,
    ),
    tf.keras.callbacks.CSVLogger(
        filename=os.path.join(_tensorboard_dir, "..", "training.log"),
        append=True,
    ),
    MemoryCallback(),
]


for index, these_files in df_input.iterrows():

    try:
        data = load_volume(these_files['t1'])
        target = load_volume(these_files['brainmask'], dtype=TARGET_DTYPE)

        data = _preprocess_data(data)
        target = _preprocess_target(target)
    except Exception:
        pass

    # Retry...
    try:
        data = load_volume(these_files['t1'])
        target = load_volume(these_files['brainmask'], dtype=TARGET_DTYPE)

        data = _preprocess_data(data)
        target = _preprocess_target(target)
    except Exception:
        with open("bad-pairs.txt", 'w') as fp:
            print(these_files['t1'], these_files['brainmask'], file=fp)

    model.fit(
        x=data,
        y=target,
        epochs=1,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=callbacks
    )

