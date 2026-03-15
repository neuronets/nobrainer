"""Unit tests for the autoresearch run_loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nobrainer.research.loop import (
    ExperimentResult,
    _classify_failure,
    _has_nan,
    _parse_config_comment,
    _patch_config,
    _read_val_dice,
    _write_summary,
    run_loop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_train_script(path: Path, config: dict | None = None) -> None:
    cfg = config or {"learning_rate": 1e-4, "batch_size": 4}
    path.write_text(
        f"# CONFIG: {json.dumps(cfg)}\n"
        "import sys; print('training done'); sys.exit(0)\n"
    )


def _write_val_dice(path: Path, val_dice: float) -> None:
    path.write_text(json.dumps({"val_dice": val_dice}))


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_parse_config_comment(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("# CONFIG: {\"lr\": 1e-4}\nprint('hi')\n")
        config = _parse_config_comment(script)
        assert config["lr"] == pytest.approx(1e-4)

    def test_parse_config_comment_missing(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("print('no config')\n")
        config = _parse_config_comment(script)
        assert config == {}

    def test_patch_config(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("# CONFIG: {\"lr\": 1e-4}\nprint('hi')\n")
        _patch_config(script, {"lr": 5e-4})
        content = script.read_text()
        assert (
            '"lr": 0.0005' in content or '"lr": 5e-4' in content or "5e-04" in content
        )

    def test_patch_config_adds_when_missing(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("print('no config')\n")
        _patch_config(script, {"lr": 1e-3})
        content = script.read_text()
        assert "# CONFIG:" in content

    def test_read_val_dice_valid(self, tmp_path):
        (tmp_path / "val_dice.json").write_text('{"val_dice": 0.85}')
        assert _read_val_dice(tmp_path / "val_dice.json") == pytest.approx(0.85)

    def test_read_val_dice_missing(self, tmp_path):
        assert _read_val_dice(tmp_path / "nonexistent.json") is None

    def test_has_nan(self):
        assert _has_nan("loss: nan after epoch 3")
        assert not _has_nan("loss: 0.25")

    def test_classify_failure_oom(self):
        assert _classify_failure("CUDA out of memory") == "CUDA OOM"

    def test_classify_failure_nan(self):
        assert _classify_failure("nan in grad") == "NaN in loss"

    def test_classify_failure_generic(self):
        assert _classify_failure("some error") == "non-zero exit code"

    def test_write_summary(self, tmp_path):
        results = [
            ExperimentResult(0, {}, 0.8, "improved"),
            ExperimentResult(1, {}, 0.79, "degraded"),
        ]
        _write_summary(tmp_path, results, "bayesian_vnet", 0.8)
        summary = (tmp_path / "run_summary.md").read_text()
        assert "bayesian_vnet" in summary
        assert "0.8000" in summary


# ---------------------------------------------------------------------------
# run_loop integration tests (subprocess mocked)
# ---------------------------------------------------------------------------


class TestRunLoop:
    def test_keep_improved_experiment(self, tmp_path):
        """run_loop keeps config when val_dice improves."""
        _write_train_script(tmp_path / "train.py")
        _write_val_dice(tmp_path / "val_dice.json", 0.9)

        with patch(
            "nobrainer.research.loop._propose_config",
            side_effect=[
                {"learning_rate": 5e-4, "batch_size": 4},
            ]
            * 5,
        ), patch(
            "nobrainer.research.loop.subprocess.run",
        ) as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "training done\n"
            mock_run.return_value.stderr = ""
            results = run_loop(
                tmp_path,
                max_experiments=2,
                budget_hours=1.0,
            )

        improved = [r for r in results if r.outcome == "improved"]
        assert len(improved) >= 1

    def test_revert_on_degraded(self, tmp_path):
        """run_loop reverts train.py when val_dice degrades."""
        original_content = (
            f"# CONFIG: {json.dumps({'learning_rate': 1e-4, 'batch_size': 4})}\n"
            "import sys; sys.exit(0)\n"
        )
        (tmp_path / "train.py").write_text(original_content)
        # First experiment improves, second degrades
        dices = [0.8, 0.7]

        call_count = [0]

        def _mock_run(cmd, **kwargs):
            from unittest.mock import MagicMock

            dice = dices[call_count[0] % len(dices)]
            _write_val_dice(tmp_path / "val_dice.json", dice)
            call_count[0] += 1
            r = MagicMock()
            r.returncode = 0
            r.stdout = "done\n"
            r.stderr = ""
            return r

        with patch(
            "nobrainer.research.loop.subprocess.run", side_effect=_mock_run
        ), patch(
            "nobrainer.research.loop._propose_config",
            side_effect=[{"learning_rate": 5e-4}] * 5,
        ):
            results = run_loop(tmp_path, max_experiments=2, budget_hours=1.0)

        degraded = [r for r in results if r.outcome == "degraded"]
        assert len(degraded) >= 1

    def test_failure_handling_reverts(self, tmp_path):
        """run_loop reverts train.py when subprocess fails."""
        _write_train_script(tmp_path / "train.py")
        original = (tmp_path / "train.py").read_text()

        with patch(
            "nobrainer.research.loop.subprocess.run",
        ) as mock_run, patch(
            "nobrainer.research.loop._propose_config",
            return_value={"learning_rate": 1e-3},
        ):
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "some error"
            results = run_loop(tmp_path, max_experiments=1, budget_hours=1.0)

        assert results[0].outcome == "failed"
        # Train script reverted
        assert (tmp_path / "train.py").read_text() == original

    def test_run_summary_written(self, tmp_path):
        """run_summary.md is written after the loop."""
        _write_train_script(tmp_path / "train.py")
        with patch(
            "nobrainer.research.loop.subprocess.run",
        ) as mock_run, patch(
            "nobrainer.research.loop._propose_config",
            return_value={"learning_rate": 1e-4},
        ):
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "error"
            run_loop(tmp_path, max_experiments=1, budget_hours=1.0)

        assert (tmp_path / "run_summary.md").exists()

    def test_missing_train_script_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_loop(tmp_path, max_experiments=1, budget_hours=1.0)

    def test_budget_seconds_terminates_quickly(self, tmp_path):
        """T013: budget_seconds=10 should terminate within 15s."""
        import time

        (tmp_path / "train.py").write_text(
            "import json, time; time.sleep(0.1);\n"
            'json.dump({"val_dice": 0.5}, open("val_dice.json", "w"))\n'
        )
        start = time.time()
        with patch(
            "nobrainer.research.loop._propose_config",
            return_value={},
        ):
            run_loop(
                tmp_path,
                max_experiments=100,
                budget_seconds=5,
            )
        elapsed = time.time() - start
        assert elapsed < 15, f"Loop took {elapsed:.1f}s, expected < 15s"
