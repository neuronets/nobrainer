"""Integration test for commit_best_model with a real DataLad dataset.

Requirements: datalad>=0.19 and git-annex must be installed.
No OSF remote is configured — OSF push is skipped gracefully.
The 1-hour SC-008 SLA for OSF retrieval requires live OSF and is not
validated here (manual verification only).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

datalad = pytest.importorskip("datalad", reason="datalad not installed")


@pytest.fixture()
def trained_models_dataset(tmp_path):
    """Create a fresh DataLad dataset in tmp_path/trained_models."""
    import datalad.api as dl

    trained_models = tmp_path / "trained_models"
    trained_models.mkdir()
    dl.create(path=str(trained_models))
    return trained_models


@pytest.fixture()
def model_files(tmp_path):
    """Create dummy model.pth and config.json files."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "best_model.pth"
    torch.save({"weights": torch.randn(4, 4)}, str(model_path))
    config_path = run_dir / "best_config.json"
    config_path.write_text(json.dumps({"learning_rate": 1e-4, "batch_size": 4}))
    return model_path, config_path


class TestCommitBestModelIntegration:
    def test_files_committed_to_datalad(self, trained_models_dataset, model_files):
        """commit_best_model creates model.pth, config.json, model_card.md in dataset."""
        from nobrainer.research.loop import commit_best_model

        model_path, config_path = model_files
        result = commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_dataset,
            model_family="bayesian_vnet",
            val_dice=0.87,
            source_run_id="integration_test_001",
        )

        dest = Path(result["path"])
        assert (dest / "model.pth").exists()
        assert (dest / "config.json").exists()
        assert (dest / "model_card.md").exists()

    def test_datalad_dataset_is_clean_after_commit(
        self, trained_models_dataset, model_files
    ):
        """datalad status shows no untracked/modified files after commit_best_model."""
        import datalad.api as dl

        from nobrainer.research.loop import commit_best_model

        model_path, config_path = model_files
        commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_dataset,
            model_family="bayesian_vnet",
            val_dice=0.87,
        )

        status_results = list(dl.status(dataset=str(trained_models_dataset)))
        unclean = [r for r in status_results if r.get("state") not in ("clean", None)]
        assert len(unclean) == 0, f"Expected clean dataset, got: {unclean}"

    def test_git_log_contains_commit_message(self, trained_models_dataset, model_files):
        """Git log in DataLad dataset contains the autoresearch commit."""
        from nobrainer.research.loop import commit_best_model

        model_path, config_path = model_files
        result = commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_dataset,
            model_family="bayesian_vnet",
            val_dice=0.88,
        )

        git_log = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=str(trained_models_dataset),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "bayesian_vnet" in git_log.stdout
        assert "0.8800" in git_log.stdout
        assert result["datalad_commit"] in git_log.stdout

    def test_directory_structure_follows_convention(
        self, trained_models_dataset, model_files
    ):
        """Model files land under neuronets/autoresearch/<model_family>/<date>/."""
        from nobrainer.research.loop import commit_best_model

        model_path, config_path = model_files
        result = commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_dataset,
            model_family="bayesian_vnet",
            val_dice=0.90,
        )

        dest = Path(result["path"])
        # Path must be: <trained_models>/neuronets/autoresearch/bayesian_vnet/<YYYY-MM-DD>
        parts = dest.parts
        assert "neuronets" in parts
        assert "autoresearch" in parts
        assert "bayesian_vnet" in parts

    def test_model_card_contains_required_metadata(
        self, trained_models_dataset, model_files
    ):
        """model_card.md includes model family, val_dice, source_run_id, and versions."""
        from nobrainer.research.loop import commit_best_model

        model_path, config_path = model_files
        result = commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_dataset,
            model_family="bayesian_vnet",
            val_dice=0.85,
            source_run_id="run_abc123",
        )

        card = (Path(result["path"]) / "model_card.md").read_text()
        assert "bayesian_vnet" in card
        assert "0.8500" in card
        assert "run_abc123" in card
        assert "PyTorch" in card

    def test_osf_push_skipped_gracefully_when_no_remote(
        self, trained_models_dataset, model_files
    ):
        """No OSF remote configured — osf_url is None, function completes normally."""
        from nobrainer.research.loop import commit_best_model

        model_path, config_path = model_files
        result = commit_best_model(
            best_model_path=model_path,
            best_config_path=config_path,
            trained_models_path=trained_models_dataset,
            model_family="bayesian_vnet",
            val_dice=0.80,
        )

        # Without an OSF remote, push fails gracefully and osf_url is None
        assert result["osf_url"] is None
