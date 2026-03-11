"""Unit tests for commit_best_model in nobrainer.research.loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_model_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create dummy model and config files."""
    model_path = tmp_path / "best_model.pth"
    model_path.write_bytes(b"\x00" * 16)  # dummy weights
    config_path = tmp_path / "best_config.json"
    config_path.write_text(json.dumps({"learning_rate": 1e-4, "batch_size": 4}))
    return model_path, config_path


class TestCommitBestModel:
    def test_directory_structure_created(self, tmp_path):
        """commit_best_model creates the expected subdirectory."""
        model_path, config_path = _make_model_files(tmp_path)
        trained_models = tmp_path / "trained_models"
        trained_models.mkdir()

        mock_dl = MagicMock()
        mock_dl.save = MagicMock()
        mock_dl.push = MagicMock()

        with patch.dict(
            "sys.modules", {"datalad": MagicMock(), "datalad.api": mock_dl}
        ):
            from nobrainer.research.loop import commit_best_model

            result = commit_best_model(
                best_model_path=model_path,
                best_config_path=config_path,
                trained_models_path=trained_models,
                model_family="bayesian_vnet",
                val_dice=0.85,
                source_run_id="run_001",
            )

        dest = Path(result["path"])
        assert dest.exists()
        assert (dest / "model.pth").exists()
        assert (dest / "config.json").exists()
        assert (dest / "model_card.md").exists()

    def test_model_card_contains_required_fields(self, tmp_path):
        """model_card.md contains architecture, val_dice, source_run_id."""
        model_path, config_path = _make_model_files(tmp_path)
        trained_models = tmp_path / "trained_models"
        trained_models.mkdir()

        mock_dl = MagicMock()

        with patch.dict(
            "sys.modules", {"datalad": MagicMock(), "datalad.api": mock_dl}
        ):
            from nobrainer.research.loop import commit_best_model

            result = commit_best_model(
                best_model_path=model_path,
                best_config_path=config_path,
                trained_models_path=trained_models,
                model_family="bayesian_vnet",
                val_dice=0.85,
                source_run_id="run_42",
            )

        card = (Path(result["path"]) / "model_card.md").read_text()
        assert "bayesian_vnet" in card
        assert "0.8500" in card
        assert "run_42" in card
        assert "PyTorch" in card

    def test_model_version_dict_fields(self, tmp_path):
        """commit_best_model returns ModelVersion dict with expected keys."""
        model_path, config_path = _make_model_files(tmp_path)
        trained_models = tmp_path / "trained_models"
        trained_models.mkdir()

        mock_dl = MagicMock()

        with patch.dict(
            "sys.modules", {"datalad": MagicMock(), "datalad.api": mock_dl}
        ):
            from nobrainer.research.loop import commit_best_model

            result = commit_best_model(
                best_model_path=model_path,
                best_config_path=config_path,
                trained_models_path=trained_models,
                model_family="bayesian_vnet",
                val_dice=0.75,
            )

        assert "path" in result
        assert "datalad_commit" in result
        assert "val_dice" in result
        assert "model_family" in result
        assert result["val_dice"] == pytest.approx(0.75)
        assert result["model_family"] == "bayesian_vnet"

    def test_datalad_commit_message_in_result(self, tmp_path):
        """commit_best_model result contains a descriptive datalad_commit message."""
        model_path, config_path = _make_model_files(tmp_path)
        trained_models = tmp_path / "trained_models"
        trained_models.mkdir()

        mock_dl = MagicMock()

        with patch.dict(
            "sys.modules", {"datalad": MagicMock(), "datalad.api": mock_dl}
        ):
            from nobrainer.research.loop import commit_best_model

            result = commit_best_model(
                best_model_path=model_path,
                best_config_path=config_path,
                trained_models_path=trained_models,
                model_family="bayesian_vnet",
                val_dice=0.9,
            )

        assert "bayesian_vnet" in result["datalad_commit"]
        assert "0.9000" in result["datalad_commit"]

    def test_result_contains_osf_url_key(self, tmp_path):
        """commit_best_model result always contains the osf_url key."""
        model_path, config_path = _make_model_files(tmp_path)
        trained_models = tmp_path / "trained_models"
        trained_models.mkdir()

        with patch.dict(
            "sys.modules", {"datalad": MagicMock(), "datalad.api": MagicMock()}
        ):
            from nobrainer.research.loop import commit_best_model

            result = commit_best_model(
                best_model_path=model_path,
                best_config_path=config_path,
                trained_models_path=trained_models,
                model_family="bayesian_vnet",
                val_dice=0.8,
            )

        # osf_url is present; it is either 'osf://' (push succeeded) or None
        assert "osf_url" in result

    def test_datalad_not_installed_raises_import_error(self, tmp_path):
        """ImportError raised with helpful message when datalad missing."""
        model_path, config_path = _make_model_files(tmp_path)
        trained_models = tmp_path / "trained_models"
        trained_models.mkdir()

        with patch.dict("sys.modules", {"datalad": None, "datalad.api": None}):
            from nobrainer.research.loop import commit_best_model

            with pytest.raises(ImportError, match="datalad"):
                commit_best_model(
                    best_model_path=model_path,
                    best_config_path=config_path,
                    trained_models_path=trained_models,
                    model_family="bayesian_vnet",
                    val_dice=0.8,
                )
