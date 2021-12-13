import tensorflow as tf


def centercrop(x, y=None, finesize=64, trans_xy=False):
    """Crops the given image at the center.
    ...
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> finesize = 1
    >>> x_out = spatial_transforms.centercrop(x, finesize=finesize)
    >>> x_out
        <tf.Tensor: shape=(1, 1, 3), dtype=float32,
        numpy=array([[[2., 2., 2.]]], dtype=float32)>
    ...
    Args:
        Input x is a tensor or numpy to have rank 3,
        Label y is a tensor or numpy to have rank 3,
        finesize is the size of the cropped output,
        finesize (int): Desired output size of the crop. Default = 64;
        Trans_xy (Boolean): transform both x and y. If set True, function
        will require both x,y.
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
        return x, y
    else:
        return x


def spatialConstantPadding(x, y=None, trans_xy=False, padding_zyx=[1, 1, 1]):
    """Add constant padding
    ...
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
    ...
    Args:
        Input x is a tensor or numpy to have rank 3,
        Label y is a tensor or numpy to have rank 3,
        padding_zyx: Desired padding in three dimensions. Default = 1;
        Trans_xy (Boolean): transform both x and y. If set True, function
        will require both x,y.
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
        return x, y
    else:
        return x


def randomCrop(x, y=None, trans_xy=False, cropsize=16):
    """Crops the given image from random locations
    ...
    >>> x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
    >>> x_out = spatial_transforms.randomCrop(x, cropsize=1)
    >>> x_out
    <tf.Tensor: shape=(1, 1, 3), dtype=float32,
    numpy=array([[[6., 2., 5.]]], dtype=float32)>
    ...
    Args:
        Input x is a tensor or numpy to have rank 3,
        Label y is a tensor or numpy to have rank 3,
        cropsize is the size of the cropped output,
        finesize (int): Desired output size of the crop. Default = 64;
        Trans_xy (Boolean): transform both x and y. If set True, function
        will require both x,y.
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
    else:
        return tf.image.random_crop(x, [cropsize, cropsize, x.shape[2]])


def resize(x, y=None, trans_xy=False, size=[32, 32], mode="bicubic"):
    """Resize the input and label
    ...
    >>> x = np.array([[[1, 2, 3], [6, 2, 5], [3, 4, 9]]])
    >>> x_out = spatial_transforms.resize(x,size=[2, 2])
    >>> x_out
    <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
    array([[[2.0145986, 1.9562043, 3.2919707],
            [3.678832 , 3.620438 , 8.284672 ]],

           [[2.0145986, 1.9562043, 3.2919707],
            [3.678832 , 3.620438 , 8.284672 ]]], dtype=float32)>
    ...

    Args:
        Input x is a tensor or numpy to have rank 3,
        Label y is a tensor or numpy to have rank 3,
        size: the resize output
        mode: bilinear, lanczos3, lanczos5, bicubic, gaussian , nearest.
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
        return x, y
    else:
        return x


def randomflip_leftright(x, y=None, trans_xy=False):
    """Randomly flips the input and label randomly with a given probability
    ...
    >>> x = np.array([[[1, 2, 3], [6, 2, 5], [3, 4, 9]]])
    >>> x_out = spatial_transforms.randomflip_leftright(x)
    >>> x_out
    <tf.Tensor: shape=(1, 3, 3), dtype=float32, numpy=
    array([[[3., 4., 9.],
            [6., 2., 5.],
            [1., 2., 3.]]], dtype=float32)>
    ...
    Args:
        Input x is a tensor or numpy to have rank 3,
        Label y is a tensor or numpy to have rank 3,
        trans_xy (float): Transform both x and y, default set False.
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
    else:
        return tf.image.random_flip_left_right(x, seed=None)
