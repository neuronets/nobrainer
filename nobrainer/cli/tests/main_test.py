"""Tests for `nobrainer.cli.main`."""

import csv
from pathlib import Path

from click.testing import CliRunner
import nibabel as nib
import numpy as np
import pytest

from .. import main as climain
from ...io import read_csv
from ...models.meshnet import meshnet
from ...models.progressivegan import progressivegan
from ...utils import get_data


def test_convert_nonscalar_labels(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        csvpath = get_data(tmp_path)
        tfrecords_template = Path("data/shard-{shard:03d}.tfrecords")
        tfrecords_template.parent.mkdir(exist_ok=True)
        args = """\
    convert --csv={} --tfrecords-template={} --volume-shape 256 256 256
        --examples-per-shard=2 --to-ras --no-verify-volumes
    """.format(
            csvpath, tfrecords_template
        )
        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        assert Path("data/shard-000.tfrecords").is_file()
        assert Path("data/shard-001.tfrecords").is_file()
        assert Path("data/shard-002.tfrecords").is_file()
        assert Path("data/shard-003.tfrecords").is_file()
        assert Path("data/shard-004.tfrecords").is_file()
        assert not Path("data/shard-005.tfrecords").is_file()


def test_convert_scalar_int_labels(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        csvpath = get_data(str(tmp_path))
        # Make labels scalars.
        data = [(x, 0) for (x, _) in read_csv(csvpath)]
        csvpath = tmp_path.with_suffix(".new.csv")
        with open(csvpath, "w", newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(data)
        tfrecords_template = Path("data/shard-{shard:03d}.tfrecords")
        tfrecords_template.parent.mkdir(exist_ok=True)
        args = """\
    convert --csv={} --tfrecords-template={} --volume-shape 256 256 256
        --examples-per-shard=2 --to-ras --no-verify-volumes
    """.format(
            csvpath, tfrecords_template
        )
        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        assert Path("data/shard-000.tfrecords").is_file()
        assert Path("data/shard-001.tfrecords").is_file()
        assert Path("data/shard-002.tfrecords").is_file()
        assert Path("data/shard-003.tfrecords").is_file()
        assert Path("data/shard-004.tfrecords").is_file()
        assert not Path("data/shard-005.tfrecords").is_file()


def test_convert_scalar_float_labels(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        csvpath = get_data(str(tmp_path))
        # Make labels scalars.
        data = [(x, 1.0) for (x, _) in read_csv(csvpath)]
        csvpath = tmp_path.with_suffix(".new.csv")
        with open(csvpath, "w", newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(data)
        tfrecords_template = Path("data/shard-{shard:03d}.tfrecords")
        tfrecords_template.parent.mkdir(exist_ok=True)
        args = """\
    convert --csv={} --tfrecords-template={} --volume-shape 256 256 256
        --examples-per-shard=2 --to-ras --no-verify-volumes
    """.format(
            csvpath, tfrecords_template
        )
        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        assert Path("data/shard-000.tfrecords").is_file()
        assert Path("data/shard-001.tfrecords").is_file()
        assert Path("data/shard-002.tfrecords").is_file()
        assert Path("data/shard-003.tfrecords").is_file()
        assert Path("data/shard-004.tfrecords").is_file()
        assert not Path("data/shard-005.tfrecords").is_file()


def test_convert_multi_resolution(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        csvpath = get_data(str(tmp_path))
        # Make labels scalars.
        data = [(x, 1.0) for (x, _) in read_csv(csvpath)]
        csvpath = tmp_path.with_suffix(".new.csv")
        with open(csvpath, "w", newline="") as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(data)
        tfrecords_template = Path("data/shard-{shard:03d}.tfrecords")
        tfrecords_template.parent.mkdir(exist_ok=True)
        args = """\
    convert --csv={} --tfrecords-template={} --volume-shape 256 256 256 --start-resolution 64
        --examples-per-shard=2 --no-verify-volumes --multi-resolution
    """.format(
            csvpath, tfrecords_template
        )
        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0

        resolutions = [64, 128, 256]
        for res in resolutions:
            assert Path("data/shard-000-res-{:03d}.tfrecords".format(res)).is_file()
            assert Path("data/shard-001-res-{:03d}.tfrecords".format(res)).is_file()
            assert Path("data/shard-002-res-{:03d}.tfrecords".format(res)).is_file()
            assert Path("data/shard-003-res-{:03d}.tfrecords".format(res)).is_file()
            assert Path("data/shard-004-res-{:03d}.tfrecords".format(res)).is_file()
            assert not Path("data/shard-005-res-{:03d}.tfrecords".format(res)).is_file()


@pytest.mark.xfail
def test_merge():
    assert False


def test_predict():
    runner = CliRunner()
    with runner.isolated_filesystem():
        model = meshnet(1, (10, 10, 10, 1))
        model_path = "model.h5"
        model.save(model_path)

        img_path = "features.nii.gz"
        nib.Nifti1Image(np.random.randn(20, 20, 20), np.eye(4)).to_filename(img_path)
        out_path = "predictions.nii.gz"

        args = """\
    predict --model={} --block-shape 10 10 10 --resize-features-to 20 20 20
        --largest-label --rotate-and-predict {} {}
    """.format(
            model_path, img_path, out_path
        )

        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        assert Path("predictions.nii.gz").is_file()
        assert nib.load(out_path).shape == (20, 20, 20)


def test_generate():
    runner = CliRunner()
    with runner.isolated_filesystem():
        generator, _ = progressivegan(
            latent_size=256, g_fmap_base=1024, d_fmap_base=1024
        )
        resolutions = [8, 16]
        Path("models").mkdir(exist_ok=True)
        for res in resolutions:
            generator.add_resolution()
            generator([np.random.random((1, 256)), 1.0])  # to build the model by a call
            model_path = "models/generator_res_{}".format(res)
            generator.save(model_path)
            assert Path(model_path).is_dir()

        out_path = "generated.nii.gz"

        args = """\
    generate --model {} --multi-resolution --latent-size 256 {}
    """.format(
            "models", out_path
        )
        result = runner.invoke(climain.cli, args.split())
        assert result.exit_code == 0
        for res in resolutions:
            assert Path("generated_res_{}.nii.gz".format(res)).is_file()
            assert nib.load("generated_res_{}.nii.gz".format(res)).shape == (
                res,
                res,
                res,
            )


@pytest.mark.xfail
def test_save():
    assert False


@pytest.mark.xfail
def test_evaluate():
    assert False


def test_info():
    runner = CliRunner()
    result = runner.invoke(climain.cli, ["info"])
    assert result.exit_code == 0
    assert "Python" in result.output
    assert "System" in result.output
    assert "Timestamp" in result.output
