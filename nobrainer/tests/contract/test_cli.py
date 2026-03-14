"""CLI contract tests for nobrainer commands.

Verifies that all CLI commands advertised in contracts/nobrainer-pytorch-api.md
are present, have the expected options, and exit with code 0 on --help.
"""

from __future__ import annotations

import subprocess
import sys


def _help(cmd: list[str]) -> str:
    """Run `nobrainer <cmd> --help` and return stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "nobrainer.cli.main"] + cmd + ["--help"],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"'{' '.join(cmd)} --help' exited {result.returncode}:\n{result.stderr}"
    return result.stdout


class TestPredictCommand:
    def test_predict_help_exits_zero(self):
        _help(["predict"])

    def test_predict_has_model_option(self):
        out = _help(["predict"])
        assert "--model" in out or "-m" in out

    def test_predict_has_model_type_option(self):
        out = _help(["predict"])
        assert "--model-type" in out

    def test_predict_has_n_classes_option(self):
        out = _help(["predict"])
        assert "--n-classes" in out

    def test_predict_has_device_option(self):
        out = _help(["predict"])
        assert "--device" in out

    def test_predict_has_n_samples_option(self):
        out = _help(["predict"])
        assert "--n-samples" in out


class TestGenerateCommand:
    def test_generate_help_exits_zero(self):
        _help(["generate"])

    def test_generate_has_model_option(self):
        out = _help(["generate"])
        assert "--model" in out or "-m" in out

    def test_generate_has_model_type_option(self):
        out = _help(["generate"])
        assert "--model-type" in out

    def test_generate_has_n_samples_option(self):
        out = _help(["generate"])
        assert "--n-samples" in out

    def test_generate_has_latent_size_option(self):
        out = _help(["generate"])
        assert "--latent-size" in out


class TestConvertTfrecordsCommand:
    def test_convert_tfrecords_help_exits_zero(self):
        _help(["convert-tfrecords"])

    def test_convert_tfrecords_has_input_option(self):
        out = _help(["convert-tfrecords"])
        assert "--input" in out or "-i" in out

    def test_convert_tfrecords_has_output_dir_option(self):
        out = _help(["convert-tfrecords"])
        assert "--output-dir" in out


class TestResearchCommand:
    def test_research_help_exits_zero(self):
        _help(["research"])

    def test_research_has_working_dir_option(self):
        out = _help(["research"])
        assert "--working-dir" in out

    def test_research_has_max_experiments_option(self):
        out = _help(["research"])
        assert "--max-experiments" in out

    def test_research_has_budget_hours_option(self):
        out = _help(["research"])
        assert "--budget-hours" in out


class TestCommitCommand:
    def test_commit_help_exits_zero(self):
        _help(["commit"])

    def test_commit_has_model_path_option(self):
        out = _help(["commit"])
        assert "--model-path" in out

    def test_commit_has_config_path_option(self):
        out = _help(["commit"])
        assert "--config-path" in out

    def test_commit_has_val_dice_option(self):
        out = _help(["commit"])
        assert "--val-dice" in out


class TestInfoCommand:
    def test_info_help_exits_zero(self):
        _help(["info"])
