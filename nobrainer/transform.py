"""Volumetric affine transformations implemented in TensorFlow."""

import tensorflow as tf


def warp_features_labels(features, labels, matrix, scalar_label=False):
    """Warp features and labels tensors according to affine matrix.

    Trilinear interpolation is used for features, and nearest neighbor
    interpolation is used for labels.

    Parameters
    ----------
    features: Rank 3 tensor, volumetric feature data.
    labels: Rank 3 tensor or N
    matrix: Tensor with shape `(4, 4)`, affine matrix.

    Returns
    -------
    Tuple of warped features, warped labels.
    """
    features = tf.convert_to_tensor(features)
    labels = tf.convert_to_tensor(labels)

    warped_coords = _warp_coords(matrix=matrix, volume_shape=features.shape)
    features = _trilinear_interpolation(volume=features, coords=warped_coords)
    if not scalar_label:
        labels = _nearest_neighbor_interpolation(volume=labels, coords=warped_coords)
    return (features, labels)


def warp(volume, matrix, order=1):
    """Warp volume tensor according to affine matrix.

    Parameters
    ----------
    volume: Rank 3 tensor, volume data.
    matrix: Tensor with shape `(4, 4)`, affine matrix.
    order: {0, 1}, interpolation order. 0 is nearest neighbor, and 1 is
        trilinear.

    Returns
    -------
    Tensor of warped volume data.
    """
    volume = tf.convert_to_tensor(volume)

    warped_coords = _warp_coords(matrix=matrix, volume_shape=volume.shape)
    if order == 0:
        out = _nearest_neighbor_interpolation(volume=volume, coords=warped_coords)
    elif order == 1:
        out = _trilinear_interpolation(volume=volume, coords=warped_coords)
    else:
        raise ValueError("unknown 'order'. Valid values are 0 and 1.")
    return out


def _warp_coords(matrix, volume_shape):
    """Build the coordinates for a affine transform on volumetric data.

    Parameters
    ----------
    matrix: tensor with shape (4, 4), affine matrix.
    volume_shape: tuple of length 3, shape of output volume.

    Returns
    -------
    TODO check this.
    Tensor of coordinates with shape (*volume_shape, 3).
    """
    coords = _get_coordinates(volume_shape=volume_shape)
    # Append ones to play nicely with 4x4 affine.
    coords_homogeneous = tf.concat(
        [coords, tf.ones((coords.shape[0], 1), dtype=coords.dtype)], axis=1
    )
    return (coords_homogeneous @ tf.transpose(matrix))[..., :3]


def get_affine(volume_shape, rotation=[0, 0, 0], translation=[0, 0, 0]):
    """Return 4x4 affine, which encodes rotation and translation of 3D tensors.

    Parameters
    ----------
    rotation: iterable of three numbers, the yaw, pitch, and roll,
        respectively, in radians.
    translation: iterable of three numbers, the number of voxels to translate
        in the x, y, and z directions.

    Returns
    -------
    Tensor with shape `(4, 4)` and dtype float32.
    """
    volume_shape = tf.cast(volume_shape, tf.float32)
    rotation = tf.cast(rotation, tf.float32)
    translation = tf.cast(translation, tf.float32)
    if volume_shape.shape[0] != 3:
        raise ValueError("`volume_shape` must have three values")
    if rotation.shape[0] != 3:
        raise ValueError("`rotation` must have three values")
    if translation.shape[0] != 3:
        raise ValueError("`translation` must have three values")

    # ROTATION
    # yaw
    rx = tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, tf.math.cos(rotation[0]), -tf.math.sin(rotation[0]), 0],
            [0, tf.math.sin(rotation[0]), tf.math.cos(rotation[0]), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    # pitch
    ry = tf.convert_to_tensor(
        [
            [tf.math.cos(rotation[1]), 0, tf.math.sin(rotation[1]), 0],
            [0, 1, 0, 0],
            [-tf.math.sin(rotation[1]), 0, tf.math.cos(rotation[1]), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    # roll
    rz = tf.convert_to_tensor(
        [
            [tf.math.cos(rotation[2]), -tf.math.sin(rotation[2]), 0, 0],
            [tf.math.sin(rotation[2]), tf.math.cos(rotation[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    # Rotation around origin.
    transform = rz @ ry @ rx

    center = tf.convert_to_tensor(volume_shape / 2 - 0.5, dtype=tf.float32)
    neg_center = tf.math.negative(center)
    center_to_origin = tf.convert_to_tensor(
        [
            [1, 0, 0, neg_center[0]],
            [0, 1, 0, neg_center[1]],
            [0, 0, 1, neg_center[2]],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    origin_to_center = tf.convert_to_tensor(
        [
            [1, 0, 0, center[0]],
            [0, 1, 0, center[1]],
            [0, 0, 1, center[2]],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    # Rotation around center of volume.
    transform = origin_to_center @ transform @ center_to_origin

    # TRANSLATION
    translation = tf.convert_to_tensor(
        [
            [1, 0, 0, translation[0]],
            [0, 1, 0, translation[1]],
            [0, 0, 1, translation[2]],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    transform = translation @ transform

    # REFLECTION
    #
    # TODO.
    # See http://web.iitd.ac.in/~hegde/cad/lecture/L6_3dtrans.pdf#page=7
    # and https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2

    return transform


def _get_coordinates(volume_shape):
    """Get coordinates that represent every voxel in a volume with shape
    `volume_shape`.

    Parameters
    ----------
    volume_shape: tuple of length 3, shape of output volume.

    Returns
    -------
    Tensor of coordinates with shape `(prod(volume_shape), 3)`.
    """
    if len(volume_shape) != 3:
        raise ValueError("shape must have 3 items.")
    dtype = tf.float32
    rows, cols, depth = volume_shape

    out = tf.meshgrid(
        tf.range(rows, dtype=dtype),
        tf.range(cols, dtype=dtype),
        tf.range(depth, dtype=dtype),
        indexing="ij",
    )
    return tf.reshape(tf.stack(out, axis=3), shape=(-1, 3))


def _nearest_neighbor_interpolation(volume, coords):
    """Three-dimensional nearest neighbors interpolation."""
    volume_f = _get_voxels(volume=volume, coords=tf.round(coords))
    return tf.reshape(volume_f, volume.shape)


def _trilinear_interpolation(volume, coords):
    """Trilinear interpolation.

    Implemented according to
    https://en.wikipedia.org/wiki/Trilinear_interpolation#Method
    https://github.com/Ryo-Ito/spatial_transformer_network/blob/2555e846b328e648a456f92d4c80fce2b111599e/warp.py#L137-L222
    """
    volume = tf.cast(volume, tf.float32)
    coords = tf.cast(coords, tf.float32)
    coords_floor = tf.floor(coords)

    shape = tf.shape(volume)
    xlen = shape[0]
    ylen = shape[1]
    zlen = shape[2]

    # Get lattice points. x0 is point below x, and x1 is point above x. Same for y and
    # z.
    x0 = tf.cast(coords_floor[:, 0], tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(coords_floor[:, 1], tf.int32)
    y1 = y0 + 1
    z0 = tf.cast(coords_floor[:, 2], tf.int32)
    z1 = z0 + 1

    # Clip values to the size of the volume array.
    x0 = tf.clip_by_value(x0, 0, xlen - 1)
    x1 = tf.clip_by_value(x1, 0, xlen - 1)
    y0 = tf.clip_by_value(y0, 0, ylen - 1)
    y1 = tf.clip_by_value(y1, 0, ylen - 1)
    z0 = tf.clip_by_value(z0, 0, zlen - 1)
    z1 = tf.clip_by_value(z1, 0, zlen - 1)

    # Get the indices at corners of cube.
    i000 = x0 * ylen * zlen + y0 * zlen + z0
    i001 = x0 * ylen * zlen + y0 * zlen + z1
    i010 = x0 * ylen * zlen + y1 * zlen + z0
    i011 = x0 * ylen * zlen + y1 * zlen + z1
    i100 = x1 * ylen * zlen + y0 * zlen + z0
    i101 = x1 * ylen * zlen + y0 * zlen + z1
    i110 = x1 * ylen * zlen + y1 * zlen + z0
    i111 = x1 * ylen * zlen + y1 * zlen + z1

    # Get volume values at corners of cube.
    volume_flat = tf.reshape(volume, [-1])
    c000 = tf.gather(volume_flat, i000)
    c001 = tf.gather(volume_flat, i001)
    c010 = tf.gather(volume_flat, i010)
    c011 = tf.gather(volume_flat, i011)
    c100 = tf.gather(volume_flat, i100)
    c101 = tf.gather(volume_flat, i101)
    c110 = tf.gather(volume_flat, i110)
    c111 = tf.gather(volume_flat, i111)

    xd = coords[:, 0] - tf.cast(x0, tf.float32)
    yd = coords[:, 1] - tf.cast(y0, tf.float32)
    zd = coords[:, 2] - tf.cast(z0, tf.float32)

    # Interpolate along x-axis.
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y-axis.
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z-axis.
    c = c0 * (1 - zd) + c1 * zd

    return tf.reshape(c, volume.shape)


def _get_voxels(volume, coords):
    """Get voxels from volume at points. These voxels are in a flat tensor."""
    x = tf.cast(volume, tf.float32)
    coords = tf.cast(coords, tf.float32)

    if len(x.shape) != 3:
        raise ValueError("`volume` must be rank 3")
    if len(coords.shape) != 2 or coords.shape[1] != 3:
        raise ValueError("`coords` must have shape `(N, 3)`.")

    rows, cols, depth = x.shape

    # Points in flattened array representation.
    fcoords = coords[:, 0] * cols * depth + coords[:, 1] * depth + coords[:, 2]

    # Some computed finds are out of range of the image's flattened size.
    # Zero those so we don't get errors. These points in the volume are filled later.
    fcoords_size = tf.size(fcoords, out_type=fcoords.dtype)
    fcoords = tf.clip_by_value(fcoords, 0, fcoords_size - 1)
    xflat = tf.reshape(x, [-1])

    # Reorder image data to transformed space.
    xflat = tf.gather(params=xflat, indices=tf.cast(fcoords, tf.int32))

    # Zero image data that was out of frame.
    outofframe = (
        tf.reduce_any(coords < 0, -1)
        | (coords[:, 0] > rows)
        | (coords[:, 1] > cols)
        | (coords[:, 2] > depth)
    )
    xflat = tf.multiply(xflat, tf.cast(tf.logical_not(outofframe), xflat.dtype))

    return xflat
