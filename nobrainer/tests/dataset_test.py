import pytest

from .. import dataset


# TODO: need to implement this soon.
@pytest.mark.xfail
def test_get_dataset():
    assert False


def test_get_steps_per_epoch():
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=1,
        volume_shape=(256, 256, 256),
        block_shape=(64, 64, 64),
        batch_size=1,
    )
    assert nsteps == 64
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=1,
        volume_shape=(256, 256, 256),
        block_shape=(64, 64, 64),
        batch_size=64,
    )
    assert nsteps == 1
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=1,
        volume_shape=(256, 256, 256),
        block_shape=(64, 64, 64),
        batch_size=63,
    )
    assert nsteps == 2
    nsteps = dataset.get_steps_per_epoch(
        n_volumes=10,
        volume_shape=(256, 256, 256),
        block_shape=(128, 128, 128),
        batch_size=4,
    )
    assert nsteps == 20
