import tensorflow as tf


def centercrop(x, y=None, finesize=64, trans_xy=False):
    """Apply center crop to input and label.

    Usage:
    ```python
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> finesize = 1
    >>> x_out = spatial_transforms.centercrop(x, finesize=finesize)
    >>> x_out
        <tf.Tensor: shape=(1, 1, 3), dtype=float32,
        numpy=array([[[2., 2., 2.]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        finesize: int, desired output size of the crop. Default = 64;
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.

    Returns
    ----------
        CenterCroped input and/or label tensor.
    """
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    if len(x.shape) != 3:
        raise ValueError("`volume` must be rank 3")
    w, h = x.shape[1], x.shape[0]
    th, tw = finesize, finesize
    x1 = int(round((w - tw) / 2.0))
    y1 = int(round((h - th) / 2.0))
    x = x[y1 : y1 + th, x1 : x1 + tw, :]

    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        y = y[y1 : y1 + th, x1 : x1 + tw, :]
    if y is None:
        return x
    return x, y


def spatialConstantPadding(x, y=None, trans_xy=False, padding_zyx=[1, 1, 1]):
    """Add constant padding to input and label.

    Usage:
    ```python
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> x_out = spatial_transforms.spatialConstantPadding(
    x,padding_zyx=[0, 1, 1])
    >>> x_out
    <tf.Tensor: shape=(1, 5, 5), dtype=float32, numpy=
    array([[[0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 2., 2., 2., 0.],
            [0., 3., 3., 3., 0.],
            [0., 0., 0., 0., 0.]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        padding_zyx: int or a list of desired padding in three dimensions.
            Default = 1;
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.

    Returns
    ----------
        Input and/or label tensor with spatial padding.
    """
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    padz = padding_zyx[0]
    pady = padding_zyx[1]
    padx = padding_zyx[2]
    padding = tf.constant([[padz, padz], [pady, pady], [padx, padx]])
    x = tf.pad(x, padding, "CONSTANT")
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        y = tf.pad(y, padding, "CONSTANT")
    if y is None:
        return x
    return x, y


def randomCrop(x, y=None, trans_xy=False, cropsize=16):
    """Apply random crops to input and label.

    Usage:
    ```python
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> x_out = spatial_transforms.randomCrop(x, cropsize=1)
    >>> x_out
    <tf.Tensor: shape=(1, 1, 3), dtype=float32,
    numpy=array([[[6., 2., 5.]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        cropsize: int, the size of the cropped output,
        finesize: int, desired output size of the crop. Default = 64;
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.

    Returns
    ----------
        Randomly croped input and/or label tensor.
    """
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        stacked = tf.stack([x, y], axis=0)
        cropped = tf.image.random_crop(stacked, [2, cropsize, cropsize, x.shape[2]])
        return cropped[0], cropped[1]
    if y is None:
        return tf.image.random_crop(x, [cropsize, cropsize, x.shape[2]])
    return tf.image.random_crop(x, [cropsize, cropsize, x.shape[2]]), y


def resize(x, y=None, trans_xy=False, size=[32, 32], mode="bicubic"):
    """Resize the input and label.

    Usage:
    ```python
     >>> x = np.array([[[1, 2, 3], [6, 2, 5], [3, 4, 9]]])
     >>> x_out = spatial_transforms.resize(x,size=[2, 2])
     >>> x_out
     <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
     array([[[2.0145986, 1.9562043, 3.2919707],
             [3.678832 , 3.620438 , 8.284672 ]],

            [[2.0145986, 1.9562043, 3.2919707],
             [3.678832 , 3.620438 , 8.284672 ]]], dtype=float32)>
     ```

     Parameters
     ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        size: int or a list, the resize output,
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.
        mode [options]: "bilinear", "lanczos3",
        "lanczos5", "bicubic", "gaussian" , "nearest".

    Returns
    ----------
         Resized input and/or label tensor.
    """
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, size, method=mode)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        y = tf.image.resize(y, size, method=mode)
    if y is None:
        return x
    return x, y


def randomflip_leftright(x, y=None, trans_xy=False):
    """Randomly flips the input and label.

    Usage:
    ```python
    >>> x = np.array([[[1, 2, 3], [6, 2, 5], [3, 4, 9]]])
    >>> x_out = spatial_transforms.randomflip_leftright(x)
    >>> x_out
    <tf.Tensor: shape=(1, 3, 3), dtype=float32, numpy=
    array([[[3., 4., 9.],
            [6., 2., 5.],
            [1., 2., 3.]]], dtype=float32)>
    ```

    Parameters
    ----------
        x: input is a tensor or numpy to have rank 3,
        y: label is a tensor or numpy to have rank 3,
        trans_xy: Boolean, transforms both x and y (Default: False).
           If set True, function will require both x,y.

    Returns
    ----------
        Randomly flipped input and/or label tensor.
    """
    if ~tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    if trans_xy:
        if y is None:
            raise ValueError("`LabelMap' should be assigned")
        if len(y.shape) != 3:
            raise ValueError("`LabelMap` must be equal or higher than rank 2")
        if ~tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        y = tf.cast(y, tf.float32)
        c = tf.concat([x, y], axis=0)
        c = tf.image.random_flip_left_right(c, seed=None)
        split_channel = int(c.shape[0] / 2)
        return c[0:split_channel, :, :], c[split_channel : c.shape[0], :, :]
    if y is None:
        return tf.image.random_flip_left_right(x, seed=None)
    return tf.image.random_flip_left_right(x, seed=None), y
