# TO DO def DivisiblePad(x,y= None, trans_xy= False, k):
import numpy as np
import tensorflow as tf


def addGaussianNoise(x, y=None, trans_xy=False, noise_mean=0.0, noise_std=0.1):
    """Add Gaussian noise to input and label.

    Usage:
    ```python
    >>> x = [[[1., 1., 1.]]]
    >>> x_out = intensity_transforms.addGaussianNoise(x,
                                  noise_mean=0.0, noise_std=1)
    >>> x_out
        <tf.Tensor: shape=(1, 1, 3), dtype=float32,
        numpy=array([[[0.82689023, 1.9072294 , 1.9717102 ]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        noise_mean: int, mean of Gaussian kernel. Default = 0.0;
        noise_std: int, standard deviation of Gaussian kernel. Default=0.1;
        trans_xy: Boolean, transforms both x and y. If set True, function
        will require both x,y.

    Returns
    ----------
        Input and/or label tensor with added Gaussian noise.
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
    if y is None:
        return tf.math.add(x, noise)
    return tf.math.add(x, noise), y


def minmaxIntensityScaling(x, y=None, trans_xy=False):
    """Apply intensity scaling [0-1] to input and label.

    Usage:
    ```python
    >>> x = [[[0., 2., 1.]]]
    >>> x_out = intensity_transforms.minmaxIntensityScaling(x)
    >>> x_out
    <tf.Tensor: shape=(1, 1, 3), dtype=float32,
    numpy=array([[[0., 1. , 0.5]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        trans_xy: Boolean, transforms both x and y. If set True, function
        will require both x,y.

    Returns
    ----------
        Input and/or label tensor with scaled intensity.
    """
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
    if y is None:
        return x
    return x, y


def customIntensityScaling(x, y=None, trans_xy=False, scale_x=[0.0, 1.0], scale_y=None):
    """Apply custom intensity scaling to input and label.

    Usage:
    ```python
    >>> x = [[[2., 2., 1.]]]
    >>> y = [[[1., 0., 1.]]]
    >>> x_out, y_out = intensity_transforms.customIntensityScaling(
    x, y, trans_xy=True, scale_x=[0, 4], scale_y=[0, 3])
    >>> x_out
    <tf.Tensor: shape=(1, 1, 3), dtype=float32,
    numpy=array([[[4., 4., 0.]]], dtype=float32)>
    >>> y_out
    <tf.Tensor: shape=(1, 1, 3), dtype=float32,
    numpy=array([[[3., 0., 3.]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.
        scale_x: [minimum(int), maximum(int)]
        scale_y: [minimum(int), maximum(int)]

    Returns
    ----------
        Input and/or label tensor with custom scaled Intensity.
    """
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
    if y is None:
        return x
    return x, y


def intensityMasking(x, mask_x, y=None, trans_xy=False, mask_y=None):
    """Masking the intensity values in input and label.

    Usage:
    ```python
    >>> mask_x = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> x_out = intensity_transforms.intensityMasking(x,
                mask_x=mask_x)
    >>> x_out
    (<tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=
     array([[[0., 0., 0.],
             [0., 2., 0.],
             [0., 0., 0.]], dtype=float32)>, None)
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        mask_x: mask tensor or numpy array of same shape as x
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.

    Returns
    ----------
        Masked input and/or label tensor.
    """
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
    if y is None:
        return x
    return x, y


def contrastAdjust(x, y=None, trans_xy=False, gamma=1.0):
    """Apply contrast adjustment to input and label.

    Usage:
    ```python
    >>> gamma = 1.5
    >>> epsilon = 1e-7
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> x_out
    (<tf.Tensor: shape=(1, 3, 3), dtype=float32, numpy=
     array([[[1.       , 1.       , 1.       ],
             [1.7071067, 1.7071067, 1.7071067],
             [3.       , 3.       , 3.       ]]], dtype=float32)>, None)
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        gamma: int, a contrast adjustment constant
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.

    Returns
    ----------
        Input and/or label tensor with adjusted contrast.
    """
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
    if y is None:
        return x
    return x, y
