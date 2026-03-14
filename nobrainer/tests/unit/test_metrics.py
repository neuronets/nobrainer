"""Unit tests for nobrainer.metrics (MONAI-backed)."""

import pytest
import torch

import nobrainer.metrics as metrics_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _onehot_pair(batch=2, n_classes=2, spatial=8):
    """One-hot prediction and target tensors (B, C, D, H, W)."""
    labels = torch.randint(0, n_classes, (batch, spatial, spatial, spatial))
    y_true = torch.zeros(batch, n_classes, spatial, spatial, spatial)
    y_true.scatter_(1, labels.unsqueeze(1), 1.0)
    y_pred = torch.softmax(
        torch.randn(batch, n_classes, spatial, spatial, spatial), dim=1
    )
    # MONAI metrics expect argmax-style binary predictions; use threshold
    y_pred_bin = (y_pred == y_pred.max(dim=1, keepdim=True).values).float()
    return y_true, y_pred_bin


# ---------------------------------------------------------------------------
# dice_metric
# ---------------------------------------------------------------------------


class TestDiceMetric:
    def test_instantiation(self):
        m = metrics_module.dice_metric()
        assert m is not None

    def test_perfect_score(self):
        y, _ = _onehot_pair(batch=1, n_classes=2, spatial=8)
        m = metrics_module.dice_metric(include_background=True)
        m(y_pred=y, y=y)
        result = m.aggregate()
        assert result.item() == pytest.approx(1.0, abs=1e-4)
        m.reset()

    def test_output_scalar(self):
        y_true, y_pred = _onehot_pair()
        m = metrics_module.dice_metric()
        m(y_pred=y_pred, y=y_true)
        result = m.aggregate()
        assert result.ndim == 0 or result.numel() == 1


# ---------------------------------------------------------------------------
# jaccard_metric (MeanIoU)
# ---------------------------------------------------------------------------


class TestJaccardMetric:
    def test_instantiation(self):
        m = metrics_module.jaccard_metric()
        assert m is not None

    def test_perfect_score(self):
        y, _ = _onehot_pair(batch=1, n_classes=2, spatial=8)
        m = metrics_module.jaccard_metric()
        m(y_pred=y, y=y)
        result = m.aggregate()
        assert result.item() == pytest.approx(1.0, abs=1e-4)
        m.reset()


# ---------------------------------------------------------------------------
# hausdorff_metric
# ---------------------------------------------------------------------------


class TestHausdorffMetric:
    def test_instantiation(self):
        m = metrics_module.hausdorff_metric()
        assert m is not None

    def test_perfect_score_zero(self):
        y, _ = _onehot_pair(batch=1, n_classes=2, spatial=8)
        m = metrics_module.hausdorff_metric(include_background=False, percentile=95.0)
        m(y_pred=y, y=y)
        result = m.aggregate()
        assert result.item() == pytest.approx(0.0, abs=1e-4)
        m.reset()


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_known_metric(self):
        fn = metrics_module.get("dice")
        assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            metrics_module.get("nonexistent")
