"""GPU integration test: multi-GPU training and inference.

T035 — US4: requires 2+ GPUs. Tests DDP training speedup and
multi-GPU predict() correctness.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict
from nobrainer.training import fit


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires 2+ GPUs for multi-GPU tests",
)
class TestMultiGPU:
    def test_ddp_fit_loss_decreases(self):
        """fit() with gpus=2 produces decreasing loss."""
        torch.manual_seed(42)
        x = torch.randn(16, 1, 16, 16, 16)
        y = torch.randint(0, 2, (16, 16, 16, 16))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        model = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 2, 1)
        )

        losses = []

        def track(epoch, loss, model):
            losses.append(loss)

        result = fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters(), lr=1e-2),
            max_epochs=10,
            gpus=2,
            callbacks=[track],
        )
        assert result["final_loss"] < losses[0]

    def test_multi_gpu_predict_matches_single(self):
        """Multi-GPU predict() output matches single-GPU result."""
        torch.manual_seed(42)
        vol = np.random.rand(32, 32, 32).astype(np.float32)
        model = unet(n_classes=2)

        # Single GPU
        result_single = predict(
            inputs=vol,
            model=model,
            block_shape=(16, 16, 16),
            device="cuda:0",
        )

        # Multi GPU (auto-distributes)
        result_multi = predict(
            inputs=vol,
            model=model,
            block_shape=(16, 16, 16),
            device="cuda",
        )

        single_arr = np.asarray(result_single.dataobj)
        multi_arr = np.asarray(result_multi.dataobj)
        assert np.array_equal(single_arr, multi_arr)

    def test_ddp_speedup(self):
        """2-GPU training achieves >=1.3x speedup vs 1 GPU."""
        torch.manual_seed(42)
        x = torch.randn(32, 1, 16, 16, 16)
        y = torch.randint(0, 2, (32, 16, 16, 16))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        model = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 2, 1)
        )

        # Time single GPU
        t0 = time.time()
        fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=5,
            gpus=1,
        )
        single_time = time.time() - t0

        # Time 2 GPUs
        t0 = time.time()
        fit(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.optim.Adam(model.parameters()),
            max_epochs=5,
            gpus=2,
        )
        multi_time = time.time() - t0

        speedup = single_time / multi_time
        print(
            f"Speedup: {speedup:.2f}x (single={single_time:.1f}s, multi={multi_time:.1f}s)"
        )
        assert speedup >= 1.3, f"Speedup {speedup:.2f}x < 1.3x threshold"
