"""Unit tests for nobrainer.training.fit()."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.training import fit


def _make_loader(n=8, spatial=8, n_classes=2, batch_size=2):
    """Synthetic DataLoader for training tests."""
    x = torch.randn(n, 1, spatial, spatial, spatial)
    y = torch.randint(0, n_classes, (n, spatial, spatial, spatial))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


def _make_model(n_classes=2):
    """Tiny conv model for testing."""
    return nn.Sequential(
        nn.Conv3d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(8, n_classes, 1),
    )


class TestFit:
    def test_returns_correct_keys(self):
        model = _make_model()
        loader = _make_loader()
        result = fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=2,
        )
        assert "history" in result
        assert "checkpoint_path" in result
        assert len(result["history"]) == 2

    def test_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_model()
        loader = _make_loader()
        result = fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters(), lr=1e-2),
            max_epochs=10,
        )
        losses = [h["loss"] for h in result["history"]]
        assert (
            losses[-1] < losses[0]
        ), f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"

    def test_checkpoint_created(self, tmp_path):
        model = _make_model()
        loader = _make_loader()
        result = fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=2,
            checkpoint_dir=tmp_path,
        )
        assert result["checkpoint_path"] is not None
        assert (tmp_path / "best_model.pth").exists()
        assert (tmp_path / "croissant.json").exists()

    def test_checkpoint_croissant_content(self, tmp_path):
        """Checkpoint croissant.json contains provenance metadata."""
        import json

        model = _make_model()
        loader = _make_loader()
        fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=2,
            checkpoint_dir=tmp_path,
        )
        data = json.loads((tmp_path / "croissant.json").read_text())
        prov = data["nobrainer:provenance"]
        assert prov["epochs_trained"] > 0
        assert prov["model_architecture"] == "Sequential"
        assert prov["loss_function"] == "CrossEntropyLoss"
        assert "optimizer" in prov

    def test_epochs_completed(self):
        model = _make_model()
        loader = _make_loader()
        result = fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=3,
        )
        assert len(result["history"]) == 3

    def test_dict_batch_format(self):
        """fit() works with dict-style batches (from MONAI DataLoader)."""
        x = torch.randn(4, 1, 8, 8, 8)
        y = torch.randint(0, 2, (4, 8, 8, 8))

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {"image": x[idx], "label": y[idx]}

        loader = DataLoader(DictDataset(), batch_size=2)
        model = _make_model()
        result = fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=1,
        )
        assert len(result["history"]) == 1
