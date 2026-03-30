"""Unit tests for nobrainer.slurm utilities."""

from __future__ import annotations

import torch

from nobrainer.slurm import SlurmPreemptionHandler, load_checkpoint, save_checkpoint


class TestSlurmPreemptionHandler:
    def test_initial_state(self):
        h = SlurmPreemptionHandler()
        assert h.preempted is False

    def test_is_slurm_job(self):
        # In test environment, should be False
        assert isinstance(SlurmPreemptionHandler.is_slurm_job(), bool)


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        metrics = {"train_losses": [1.0, 0.5], "best_loss": 0.5}
        save_checkpoint(tmp_path, model, opt, epoch=5, metrics=metrics)

        assert (tmp_path / "checkpoint.pt").exists()
        assert (tmp_path / "checkpoint_meta.json").exists()

        model2 = torch.nn.Linear(4, 2)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        start, restored = load_checkpoint(tmp_path, model2, opt2)

        assert start == 6  # next epoch
        assert restored["best_loss"] == 0.5
        assert len(restored["train_losses"]) == 2

    def test_load_no_checkpoint(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        start, metrics = load_checkpoint(tmp_path, model)
        assert start == 0
        assert metrics == {}

    def test_model_weights_restored(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        model.weight.data.fill_(42.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(tmp_path, model, opt, epoch=0)

        model2 = torch.nn.Linear(4, 2)
        load_checkpoint(tmp_path, model2)
        assert torch.allclose(model2.weight.data, torch.tensor(42.0))
