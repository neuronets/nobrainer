"""Unit tests for nobrainer.experiment tracking."""

from __future__ import annotations

import json

from nobrainer.experiment import ExperimentTracker


class TestExperimentTracker:
    def test_local_logging(self, tmp_path):
        tracker = ExperimentTracker(
            output_dir=tmp_path, config={"lr": 0.001}, use_wandb=False
        )
        tracker.log({"epoch": 1, "loss": 0.5})
        tracker.log({"epoch": 2, "loss": 0.3})
        tracker.finish()

        # Check JSONL
        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["loss"] == 0.5

        # Check CSV
        csv_lines = (tmp_path / "metrics.csv").read_text().strip().split("\n")
        assert len(csv_lines) == 3  # header + 2 rows
        assert "epoch" in csv_lines[0]

        # Check config
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["lr"] == 0.001

    def test_callback(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path, use_wandb=False)
        cb = tracker.callback(variant="test")

        # Simulate training callback
        cb(0, {"loss": 1.5}, None)  # (epoch, logs_dict, model)
        cb(1, {"loss": 0.8}, None)
        tracker.finish()

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert row["epoch"] == 0
        assert row["loss"] == 1.5
        assert row["variant"] == "test"

    def test_no_wandb_by_default(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path)
        # Should not fail even without wandb installed
        tracker.log({"x": 1})
        tracker.finish()
