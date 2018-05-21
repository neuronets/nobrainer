# -*- coding: utf-8 -*-
"""Main command-line interface to nobrainer."""

from pathlib import Path

from nobrainer.cli import main
from nobrainer.io import read_csv
from nobrainer.io import read_volume
from nobrainer.testing import csv_of_volumes


def test_cli(csv_of_volumes):
    model_dir = "/tmp/tmpmodeldir"
    cmd = """train
--n-classes=2
--model=highres3dnet
--model-dir={model_dir}
--optimizer=Adam
--learning-rate=0.001
--batch-size=2
--prefetch=1
--volume-shape 8 8 8
--block-shape 8 8 8
--strides 8 8 8
--csv={filepath}
--binarize
--flip
--rotate
--gaussian
--reduce-contrast
--salt-and-pepper
    """
    cmd = cmd.replace('\n', ' ').format(
        model_dir=model_dir, filepath=csv_of_volumes).split()
    main(args=cmd)
    assert Path(model_dir).is_dir()

    save_dir = "/tmp/tmpmodeldir/savedmodel"
    cmd = """save
--model=highres3dnet
--model-dir={model_dir}
--n-classes=2
--block-shape 8 8 8
{save_dir}
    """
    cmd = cmd.replace('\n', ' ').format(
        model_dir=model_dir, save_dir=save_dir).split()
    main(args=cmd)
    assert Path(save_dir).is_dir()

    save_dir = next(Path(save_dir).glob('**/saved_model.pb'))
    input_ = read_csv(csv_of_volumes)[0][0]
    output = "/tmp/output.nii.gz"
    cmd = """predict
--block-shape 8 8 8
--model={save_dir}
{input}
{output}
    """
    cmd = cmd.replace('\n', ' ').format(
        save_dir=save_dir, input=input_, output=output).split()
    main(cmd)
    read_volume(output)
