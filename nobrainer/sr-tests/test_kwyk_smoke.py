"""Smoke tests for the kwyk reproduction pipeline.

Tests train a tiny MeshNet and Bayesian MeshNet for 1 epoch each to verify
the end-to-end pipeline works (loss is finite, prediction produces valid
NIfTI output, warm-start transfers weights correctly).
"""

import nibabel as nib
import numpy as np
import pytest
import torch

pyro = pytest.importorskip("pyro")

from nobrainer.models import get as get_model  # noqa: E402
from nobrainer.models.bayesian.warmstart import (  # noqa: E402
    warmstart_bayesian_from_deterministic,
)
from nobrainer.processing import Dataset, Segmentation  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants for tiny model
# ---------------------------------------------------------------------------
FILTERS = 16
BLOCK_SHAPE = (16, 16, 16)
N_CLASSES = 2
BATCH_SIZE = 2
MODEL_ARGS = {
    "n_classes": N_CLASSES,
    "filters": FILTERS,
    "receptive_field": 37,
    "dropout_rate": 0.25,
}


def _build_dataset(sample_data):
    """Build a small binarized Dataset from sample_data fixture."""
    # Use first 5 volumes
    pairs = sample_data[:5]
    ds = (
        Dataset.from_files(pairs, block_shape=BLOCK_SHAPE, n_classes=N_CLASSES)
        .batch(BATCH_SIZE)
        .binarize()
    )
    return ds


def _plot_learning_curve(losses, output_path):
    """Save a simple learning curve figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(losses) + 1), losses, "b-o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Smoke Test Learning Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=72, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.gpu
class TestKwykSmoke:
    """Smoke tests for the kwyk reproduction pipeline."""

    def test_deterministic_meshnet_train(self, sample_data, tmp_path):
        """Train deterministic MeshNet for 1 epoch; assert loss is finite."""
        ds = _build_dataset(sample_data)

        seg = Segmentation(
            base_model="meshnet",
            model_args={k: v for k, v in MODEL_ARGS.items() if k != "n_classes"},
        )

        # Collect losses via callback
        losses = []

        def _on_epoch(epoch, loss, model):
            losses.append(loss)

        seg.fit(ds, epochs=1, callbacks=[_on_epoch])

        assert len(losses) >= 1, "Expected at least 1 epoch of training"
        for loss_val in losses:
            assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"

        # Save learning curve
        _plot_learning_curve(losses, tmp_path / "det_learning_curve.png")
        assert (tmp_path / "det_learning_curve.png").exists()

    def test_bayesian_warmstart_train(self, sample_data, tmp_path):
        """Warm-start BayesianMeshNet from deterministic, train 1 epoch."""
        ds = _build_dataset(sample_data)

        # First train a deterministic model
        det_model = get_model("meshnet")(**MODEL_ARGS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        det_model = det_model.to(device)
        det_model.train()

        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(det_model.parameters(), lr=1e-3)

        # Quick 1-epoch train of deterministic model
        loader = ds.dataloader
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer.zero_grad()
            pred = det_model(images)
            loss = ce_loss(pred, labels)
            loss.backward()
            optimizer.step()
            break  # Just one batch for speed

        # Build Bayesian model and warm-start
        bayes_model = get_model("bayesian_meshnet")(**MODEL_ARGS)
        n_transferred = warmstart_bayesian_from_deterministic(
            bayes_model, det_model, initial_rho=-3.0
        )
        assert n_transferred > 0, "Expected at least 1 layer transferred"

        # Train Bayesian for 1 epoch
        from nobrainer.models.bayesian.utils import accumulate_kl

        bayes_model = bayes_model.to(device)
        bayes_model.train()
        optimizer_b = torch.optim.Adam(bayes_model.parameters(), lr=1e-3)

        losses = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

            if labels.ndim == images.ndim and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype in (torch.float32, torch.float64):
                labels = labels.long()

            optimizer_b.zero_grad()
            pred = bayes_model(images)
            loss = ce_loss(pred, labels) + accumulate_kl(bayes_model)
            loss.backward()
            optimizer_b.step()
            losses.append(loss.item())
            break  # Just one batch for speed

        assert len(losses) >= 1
        for loss_val in losses:
            assert np.isfinite(loss_val), f"Bayesian loss not finite: {loss_val}"

        # Save learning curve
        _plot_learning_curve(losses, tmp_path / "bayes_learning_curve.png")
        assert (tmp_path / "bayes_learning_curve.png").exists()

    def test_predict_output(self, sample_data, tmp_path):
        """Predict on 1 volume; assert NIfTI output with matching shape and Dice > 0."""
        ds = _build_dataset(sample_data)

        seg = Segmentation(
            base_model="meshnet",
            model_args={k: v for k, v in MODEL_ARGS.items() if k != "n_classes"},
        )
        seg.fit(ds, epochs=1)

        # Predict on first volume
        eval_img_path = sample_data[0][0]
        eval_lbl_path = sample_data[0][1]

        result = seg.predict(eval_img_path, block_shape=BLOCK_SHAPE)

        # Check output is NIfTI
        assert isinstance(
            result, nib.Nifti1Image
        ), f"Expected Nifti1Image, got {type(result)}"

        # Check shape matches input spatial dims
        input_img = nib.load(eval_img_path)
        input_shape = input_img.shape[:3]
        result_shape = result.shape[:3]
        assert (
            result_shape == input_shape
        ), f"Shape mismatch: input={input_shape}, output={result_shape}"

        # Check Dice > 0 against binarized ground truth
        gt_arr = np.asarray(nib.load(eval_lbl_path).dataobj, dtype=np.float32)
        gt_binary = (gt_arr > 0).astype(np.float32)

        pred_arr = np.asarray(result.dataobj, dtype=np.float32)
        pred_binary = (pred_arr > 0).astype(np.float32)

        intersection = np.logical_and(pred_binary, gt_binary).sum()
        total = pred_binary.sum() + gt_binary.sum()
        if total > 0:
            dice = float(2.0 * intersection / total)
        else:
            dice = 1.0

        assert dice > 0, f"Expected Dice > 0, got {dice}"

        # Save learning curve figure
        _plot_learning_curve([0.5], tmp_path / "predict_learning_curve.png")
        assert (tmp_path / "predict_learning_curve.png").exists()
