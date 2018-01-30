"""Scoring functions."""

import tensorflow.keras.backend as K


# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/27c47a8f38f11e446e33465146c8eb6074872678/train.py#L23-L27
def dice_coefficient(y_true, y_pred, smooth=1.):
    """Return Dice score given two boolean ndarrays."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (
        (2. * intersection + smooth)
        / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )
