"""Integration test for autoresearch loop with budget-minutes constraint.

T014: Run the full research loop with a 60-second budget and verify
it produces a run_summary.md with at least 1 experiment entry.
"""

from __future__ import annotations

from nobrainer.research.loop import run_loop


class TestResearchSmoke:
    def test_research_loop_completes_with_budget_seconds(self, tmp_path):
        """Full research loop with 60s budget, tiny MeshNet on synthetic data."""
        # Create a minimal train script that writes val_dice.json quickly
        train_script = tmp_path / "train.py"
        train_script.write_text(
            "import json, random, time\n"
            "time.sleep(0.5)\n"
            'json.dump({"val_dice": round(random.uniform(0.4, 0.9), 4)}, '
            'open("val_dice.json", "w"))\n'
        )

        results = run_loop(
            working_dir=tmp_path,
            model_family="meshnet",
            max_experiments=2,
            budget_seconds=60,
        )

        # Verify at least 1 experiment ran
        assert len(results) >= 1, "Expected at least 1 experiment"

        # Verify run_summary.md exists
        summary = tmp_path / "run_summary.md"
        assert summary.exists(), "run_summary.md not created"
        content = summary.read_text()
        assert "val_dice" in content.lower() or "experiment" in content.lower()
