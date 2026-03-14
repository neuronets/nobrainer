"""GPU end-to-end test: train a UNet on synthetic data, then verify predict()
produces high Dice on the same data (overfitting test).

T031 — US1 acceptance scenario 2: Dice >= 0.95 on a known volume.

Since we don't ship reference weights in the repo, this test creates a
synthetic brain-like volume (sphere label), trains a UNet to overfit it,
then runs predict() and checks the Dice score.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn

from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict


def _make_sphere_volume(shape=(64, 64, 64), radius=20):
    """Create a synthetic volume with a centered sphere as the label."""
    vol = np.random.rand(*shape).astype(np.float32) * 0.3
    label = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) / 2
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    mask = dist < radius
    label[mask] = 1.0
    vol[mask] += 0.7  # make sphere brighter
    return vol, label


@pytest.mark.gpu
class TestPredictEndToEnd:
    def test_unet_overfit_dice_above_threshold(self):
        """Train 2-class UNet to overfit a sphere, then check Dice >= 0.95."""
        device = torch.device("cuda")
        torch.manual_seed(42)

        vol, label = _make_sphere_volume(shape=(64, 64, 64), radius=20)
        x = torch.from_numpy(vol[None, None]).to(device)  # (1, 1, 64, 64, 64)
        # One-hot encode label for 2-class: background + foreground
        label_long = torch.from_numpy(label).long().to(device)  # (64, 64, 64)
        y_onehot = nn.functional.one_hot(label_long, 2)  # (64,64,64,2)
        y_onehot = y_onehot.permute(3, 0, 1, 2).unsqueeze(0).float()  # (1,2,64,64,64)

        model = unet(n_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Overfit on a single sample
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            pred = model(x)  # (1, 2, 64, 64, 64)
            loss = criterion(pred, label_long.unsqueeze(0))
            loss.backward()
            optimizer.step()

        # Run predict() on the same volume — returns argmax labels
        model.eval()
        result = predict(
            inputs=vol,
            model=model,
            block_shape=(32, 32, 32),
            batch_size=8,
            device="cuda",
            return_labels=True,
        )

        pred_arr = np.asarray(result.dataobj)
        # Compute Dice for class 1 (foreground)
        pred_bin = (pred_arr == 1).astype(np.float32)
        intersection = (pred_bin * label).sum()
        dice = 2 * intersection / (pred_bin.sum() + label.sum() + 1e-8)

        assert dice >= 0.95, f"Dice {dice:.4f} < 0.95 threshold"

    def test_predict_output_is_nifti_on_gpu(self):
        """Verify predict() returns a NIfTI image when run on GPU."""
        vol, _ = _make_sphere_volume(shape=(32, 32, 32))
        model = unet(n_classes=2)
        result = predict(
            inputs=vol,
            model=model,
            block_shape=(32, 32, 32),
            batch_size=1,
            device="cuda",
        )
        assert isinstance(result, nib.Nifti1Image)
        assert result.shape == (32, 32, 32)
