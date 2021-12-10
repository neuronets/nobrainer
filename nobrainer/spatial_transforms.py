import tensorflow as tf


def centercrop(x, y=None, finesize=64, trans_xy=False):
    """
    Crops the given image at the center.
    Input x is a tensor or numpy to have rank 3,
    Label y is a tensor or numpy to have rank 3,
    finesize is the size of the cropped output,
    If x is smaller than finesize along any edge, image is padded with 0
    and then center cropped.

    Args:
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
    """
    Constant padding
    Input x is a tensor or numpy to have rank 3,
    Label y is a tensor or numpy to have rank 3,
    Args:
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
    """
    Crops the given image from random locations
    Input x is a tensor or numpy to have rank 3,
    Label y is a tensor or numpy to have rank 3,
    cropsize is the size of the cropped output,
    If x is smaller than cropsize along any edge, image is padded with 0
    and then cropped.

    Args:
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
    """check image.ResizeMethod enum, or the string equivalent. Options:
    bilinear, lanczos3, lanczos5, bicubic, gaussian , nearest.
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
    """
    Randomly flips the input and label randomly with a given probability.
    x and y are Tensors or numpy arrays,
    it is expected to be of rank 3

    Args:
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
