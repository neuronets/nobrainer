"""Tests for `nobrainer.cli.main`."""

from pathlib import Path

import click
from click.testing import CliRunner
import nibabel as nib
import numpy as np
import pytest

from nobrainer.cli import main as climain
from nobrainer.io import convert
from nobrainer.io import read_csv
from nobrainer.io import read_volume
from nobrainer.models.meshnet import meshnet
from nobrainer.utils import get_data


def test_convert(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        csvpath = get_data(str(tmp_path))
        tfrecords_template = Path('data/shard-{shard:03d}.tfrecords')
        tfrecords_template.parent.mkdir(exist_ok=True)
        args = """\
    convert --csv={} --tfrecords-template={} --volume-shape 256 256 256
        --volumes-per-shard=2 --to-ras --no-verify-volumes
    """.format(csvpath, tfrecords_template)
        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        assert Path('data/shard-000.tfrecords').is_file()
        assert Path('data/shard-001.tfrecords').is_file()
        assert Path('data/shard-002.tfrecords').is_file()
        assert Path('data/shard-003.tfrecords').is_file()
        assert Path('data/shard-004.tfrecords').is_file()
        assert not Path('data/shard-005.tfrecords').is_file()


@pytest.mark.xfail
def test_merge():
    assert False


def test_predict():
    runner = CliRunner()
    with runner.isolated_filesystem():
        model = meshnet(1, (10, 10, 10, 1))
        model_path = 'model.h5'
        model.save(model_path)

        img_path = 'features.nii.gz'
        nib.Nifti1Image(np.random.randn(20, 20, 20), np.eye(4)).to_filename(img_path)
        out_path = 'predictions.nii.gz'

        args = """\
    predict --model={} --block-shape 10 10 10 --resize-features-to 20 20 20
        --largest-label --rotate-and-predict {} {}
    """.format(model_path, img_path, out_path)

        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        assert Path('predictions.nii.gz').is_file()
        assert nib.load(out_path).shape == (20, 20, 20)


@pytest.mark.xfail
def test_save():
    assert False


def test_train():
    runner = CliRunner()
    with runner.isolated_filesystem():
        xpath = 'features.nii.gz'
        ypath = 'labels.nii.gz'
        nib.Nifti1Image(np.ones((20, 20, 20)), np.eye(4)).to_filename(str(xpath))
        nib.Nifti1Image(np.ones((20, 20, 20)), np.eye(4)).to_filename(str(ypath))
        files = [(str(xpath), str(ypath))]
        convert(files, tfrecords_template='data-{shard:03d}.tf')

        args = """\
    train --model=meshnet --tfrecords-pattern={} --n-classes=1 --batch-size=1
        --volume-shape 20 20 20 --block-shape 10 10 10 --n-epochs=1 --n-volumes=1
        --loss=dice --learning-rate=1e-05
    """.format('data-*.tf')

        result = runner.invoke(climain.cli, args.split())
        print(result.output)
        assert result.exit_code == 0
        assert Path('logs').is_dir()
        assert Path('checkpoints').is_dir()


@pytest.mark.xfail
def test_evaluate():
    assert False
