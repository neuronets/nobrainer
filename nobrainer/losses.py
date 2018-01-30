"""Loss functions."""

from nobrainer.scores import dice_coefficient


def dice_loss(y_true, y_pred, smooth=1):
    """Return Dice loss given two boolean ndarrays."""
    return 1 - dice_coefficient(y_true, y_pred, smooth)
