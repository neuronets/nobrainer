import numpy as np
import pytest

from nobrainer import spatial_transforms as transformations


@pytest.fixture(scope="session")
def test_centercrop():
    # Test for inputs
    shape = (10, 10, 10)
    x = np.ones(shape).astype(np.float32)
    y = np.random.randint(0, 2, size=shape).astype(np.float32)
    fine = int(x.shape[1])
    x = transformations.centercrop(x, finesize=fine)
    x = x.numpy()
    # Test for output shapes
    assert x.shape[1] == fine & x.shape[0] == fine & x.shape[2] == shape[2]
    assert y.shape[1] == fine & y.shape[0] == fine & y.shape[2] == shape[2]

    # test for both x and y
    shape = (10, 10, 10)
    x = np.ones(shape).astype(np.float32)
    y = np.random.randint(0, 2, size=shape).astype(np.float32)
    fine = int(x.shape[1])
    x, y = transformations.centercrop(x, y, fine, trans_xy=True)
    x = x.numpy()
    y = y.numpy()
    # Test for output shapes
    assert x.shape[1] == fine & x.shape[0] == fine & x.shape[2] == shape[2]
    assert y.shape[1] == fine & y.shape[0] == fine & y.shape[2] == shape[2]

    # Test for varying finesize
    shape = (10, 10, 10)
    x = np.ones(shape).astype(np.float32)
    y = np.random.randint(0, 2, size=shape).astype(np.float32)
    finesize = [128, 1]
    x1, y1 = transformations.centercrop(x, y, finesize[0], trans_xy=True)
    x2, y2 = transformations.centercrop(x, y, finesize[1], trans_xy=True)
    x1 = x1.numpy()
    x2 = x2.numpy()
    y1 = y1.numpy()
    y2 = y2.numpy()
    assert (
        x1.shape[1]
        == min(shape[1], finesize[0]) & x1.shape[0]
        == min(shape[0], finesize[0]) & x1.shape[2]
        == shape[2]
    )
    assert (
        y1.shape[1]
        == min(shape[1], finesize[0]) & y1.shape[0]
        == min(shape[0], finesize[0]) & y1.shape[2]
        == shape[2]
    )

    assert (
        x2.shape[1]
        == min(shape[1], finesize[1]) & x2.shape[0]
        == min(shape[0], finesize[1])
    )
    assert (
        y2.shape[1]
        == min(shape[1], finesize[1]) & y2.shape[0]
        == min(shape[0], finesize[1])
    )
    assert y2.shape[2] == shape[2] & x2.shape[2] == shape[2]


def test_spatialConstantPadding():
    x = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]])
    y = np.array([[[1, 0, 1], [0, 2, 2], [3, 3, 0]], [[4, 1, 4], [5, 0, 0], [0, 0, 0]]])
    x_expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    y_expected = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 1.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    resultx, resulty = transformations.spatialConstantPadding(
        x, y, trans_xy=True, padding_zyx=[0, 2, 2]
    )
    np.testing.assert_allclose(x_expected, resultx.numpy())
    np.testing.assert_allclose(y_expected, resulty.numpy())


def test_randomCrop():
    x = np.random.rand(10, 10, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.float32)
    expected_shape = (3, 3, 10)
    res_x, res_y = transformations.randomCrop(x, y, trans_xy=True, cropsize=3)
    assert np.shape(res_x.numpy()) == expected_shape
    assert np.shape(res_y.numpy()) == expected_shape
    assert np.all(np.in1d(np.ravel(res_x), np.ravel(x)))
    assert np.all(np.in1d(np.ravel(res_y), np.ravel(y)))


def test_resize():
    x = np.random.rand(10, 10, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(10, 10, 10)).astype(np.float32)
    expected_shape = (5, 5, 10)
    results_x, results_y = transformations.resize(
        x, y, trans_xy=True, size=[5, 5], mode="bicubic"
    )
    assert np.shape(results_x.numpy()) == expected_shape
    assert np.shape(results_y.numpy()) == expected_shape


def test_randomflip_leftright():
    x = np.random.rand(3, 3, 3).astype(np.float32)
    y = np.random.randint(0, 2, size=(3, 3, 3)).astype(np.float32)
    res_x, res_y = transformations.randomflip_leftright(x, y, trans_xy=True)
    expected_shape = (3, 3, 3)
    assert np.shape(res_x.numpy()) == expected_shape
    assert np.shape(res_y.numpy()) == expected_shape
