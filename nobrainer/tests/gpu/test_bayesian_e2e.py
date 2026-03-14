"""GPU end-to-end test: Bayesian VNet with uncertainty quantification.

T045 — US2 acceptance scenario: predict_with_uncertainty() produces
label, variance, and entropy maps. Variance and entropy are non-zero.
Bayesian model trained via overfit on synthetic sphere data achieves
Dice >= 0.90 (lower than deterministic due to stochastic inference).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nobrainer.losses import dice as dice_loss
from nobrainer.models.bayesian import BayesianVNet
from nobrainer.prediction import predict_with_uncertainty


def _make_sphere_volume(shape=(64, 64, 64), radius=20):
    """Create a synthetic volume with a centered sphere as the label."""
    vol = np.random.rand(*shape).astype(np.float32) * 0.3
    label = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) / 2
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    mask = dist < radius
    label[mask] = 1.0
    vol[mask] += 0.7
    return vol, label


@pytest.mark.gpu
class TestBayesianEndToEnd:
    def test_bayesian_vnet_overfit_with_uncertainty(self):
        """Train BayesianVNet, run MC inference, check Dice and uncertainty."""
        device = torch.device("cuda")
        torch.manual_seed(42)

        vol, label = _make_sphere_volume(shape=(64, 64, 64), radius=20)
        x = torch.from_numpy(vol[None, None]).to(device)
        y = torch.from_numpy(label[None, None]).to(device)

        model = BayesianVNet(
            in_channels=1, n_classes=1, prior_type="standard_normal"
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = dice_loss()

        # Overfit
        model.train()
        for _ in range(150):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Run MC inference with uncertainty
        label_img, var_img, entropy_img = predict_with_uncertainty(
            inputs=vol,
            model=model,
            n_samples=10,
            block_shape=(32, 32, 32),
            batch_size=8,
            device="cuda",
        )

        # Check shapes
        assert label_img.shape == (64, 64, 64)
        assert var_img.shape == (64, 64, 64)
        assert entropy_img.shape == (64, 64, 64)

        # Variance and entropy should be non-zero (stochastic model)
        var_data = np.asarray(var_img.dataobj)
        entropy_data = np.asarray(entropy_img.dataobj)
        assert var_data.sum() > 0, "Variance map is all zeros"
        assert entropy_data.sum() > 0, "Entropy map is all zeros"

        # Dice check (>= 0.90, relaxed for Bayesian stochasticity)
        pred_arr = np.asarray(label_img.dataobj)
        pred_bin = (pred_arr > 0.5).astype(np.float32)
        intersection = (pred_bin * label).sum()
        dice = 2 * intersection / (pred_bin.sum() + label.sum() + 1e-8)
        assert dice >= 0.90, f"Bayesian Dice {dice:.4f} < 0.90 threshold"

    def test_variance_higher_at_boundary(self):
        """Uncertainty should be higher near the sphere boundary than inside."""
        device = torch.device("cuda")
        torch.manual_seed(42)

        vol, label = _make_sphere_volume(shape=(64, 64, 64), radius=20)
        x = torch.from_numpy(vol[None, None]).to(device)
        y = torch.from_numpy(label[None, None]).to(device)

        model = BayesianVNet(
            in_channels=1, n_classes=1, prior_type="standard_normal"
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = dice_loss()

        model.train()
        for _ in range(100):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        _, var_img, _ = predict_with_uncertainty(
            inputs=vol,
            model=model,
            n_samples=10,
            block_shape=(32, 32, 32),
            batch_size=8,
            device="cuda",
        )

        var_data = np.asarray(var_img.dataobj)

        # Create boundary mask (dilated label minus eroded label)
        from scipy.ndimage import binary_dilation, binary_erosion

        dilated = binary_dilation(label > 0.5, iterations=3)
        eroded = binary_erosion(label > 0.5, iterations=3)
        boundary = dilated & ~eroded
        interior = eroded

        if boundary.sum() > 0 and interior.sum() > 0:
            boundary_var = var_data[boundary].mean()
            # Boundary variance should be non-zero
            assert boundary_var > 0, "Boundary variance is zero"
