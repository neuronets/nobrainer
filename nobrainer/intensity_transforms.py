# TO DO def DivisiblePad(x,y= None, trans_xy= False, k):
import numpy as np
import tensorflow as tf


def addGaussianNoise(x, y=None, trans_xy=False, noise_mean=0.0, noise_std=0.1):
    """
    Adds gaussian noise to 3D tensor and label
    """
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    noise = tf.random.normal(x.shape, noise_mean, noise_std, dtype=x.dtype)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")

        y = tf.cast(y, tf.float32)
        return tf.math.add(x, noise), tf.math.add(y, noise)
    else:
        return tf.math.add(x, noise)


def minmaxIntensityScaling(x, y=None, trans_xy=False):
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    ep = tf.cast(
        tf.convert_to_tensor(1e-8 * np.ones(x.shape).astype(np.float32)), tf.float32
    )
    xmin = tf.cast(tf.reduce_min(x), tf.float32)
    xmax = tf.cast(tf.reduce_max(x), tf.float32)
    x = tf.divide(tf.subtract(x, xmin), tf.add(tf.subtract(xmax, xmin), ep))
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        ymin = tf.cast(tf.reduce_min(y), tf.float32)
        ymax = tf.cast(tf.reduce_max(y), tf.float32)
        y = tf.divide(tf.subtract(y, ymin), tf.add(tf.subtract(ymax, ymin), ep))
    return x, y


def customIntensityScaling(x, y=None, trans_xy=False, scale_x=[0.0, 1.0], scale_y=None):
    x_norm, y_norm = minmaxIntensityScaling(x, y, trans_xy)
    minx = tf.cast(
        tf.convert_to_tensor(scale_x[0] * np.ones(x_norm.shape).astype(np.float32)),
        tf.float32,
    )
    maxx = tf.cast(
        tf.convert_to_tensor(scale_x[1] * np.ones(x_norm.shape).astype(np.float32)),
        tf.float32,
    )

    diff_x = tf.subtract(maxx, minx)
    x = tf.add(tf.multiply(x_norm, diff_x), minx)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if scale_y is None:
            raise ValueError("LabelMap scaling arguments as: scale_Y=[a,b]")
        y = tf.cast(y, tf.float32)
        miny = tf.cast(
            tf.convert_to_tensor(scale_y[0] * np.ones(y_norm.shape).astype(np.float32)),
            tf.float32,
        )
        maxy = tf.cast(
            tf.convert_to_tensor(scale_y[1] * np.ones(y_norm.shape).astype(np.float32)),
            tf.float32,
        )
        diff_y = tf.subtract(maxy, miny)
        y = tf.add(tf.multiply(y_norm, diff_y), miny)
        return x, y
    else:
        return x


def intensityMasking(x, mask_x, y=None, trans_xy=False, mask_y=None):
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    if ~tf.is_tensor(mask_x):
        mask_x = tf.convert_to_tensor(mask_x)
    mask_x = tf.cast(mask_x, tf.float32)
    if mask_x.shape[0] != x.shape[0] and mask_x.shape[1] != x.shape[1]:
        raise ValueError("Masks shape should be same as Input")
    x = tf.multiply(x, mask_x)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if mask_y is None:
            raise ValueError("Label Mask should not be none")
        if ~tf.is_tensor(y) and ~tf.is_tensor(mask_y):
            y = tf.convert_to_tensor(y)
            mask_y = tf.convert_to_tensor(mask_y)
        y = tf.cast(y, tf.float32)
        mask_x = tf.cast(mask_x, tf.float32)
        if mask_y.shape[0] != y.shape[0] and mask_x.shape[1] != x.shape[1]:
            raise ValueError("Label Masks shape should be same as Label")
        return x, tf.multiply(y, mask_y)
    else:
        return x


def contrastAdjust(x, y=None, trans_xy=False, gamma=1.0):
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    ep = tf.cast(
        tf.convert_to_tensor(1e-7 * np.ones(x.shape).astype(np.float32)), tf.float32
    )
    gamma = tf.cast(
        tf.convert_to_tensor(gamma * np.ones(x.shape).astype(np.float32)), tf.float32
    )
    xmin = tf.cast(tf.reduce_min(x), tf.float32)
    xmax = tf.cast(tf.reduce_max(x), tf.float32)
    x_range = tf.subtract(xmax, xmin)
    x = tf.pow(tf.divide(tf.subtract(x, xmin), tf.add(x_range, ep)), gamma)
    x = tf.add(tf.multiply(x, x_range), xmin)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        ymin = tf.cast(tf.reduce_min(y), tf.float32)
        ymax = tf.cast(tf.reduce_max(y), tf.float32)
        y_range = tf.subtract(ymax, ymin)
        y = tf.pow(tf.divide(tf.subtract(y, ymin), tf.add(y_range, ep)), gamma)
        y = tf.add(tf.multiply(y, y_range), ymin)
        return x, y
    else:
        return x
