"""Unit tests for convert_weights() in nobrainer.io."""

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from nobrainer.io import convert_weights

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimplePT(nn.Module):
    """Minimal PyTorch model for weight-conversion tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm3d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


def _write_synthetic_h5(path: str, model: nn.Module) -> None:
    """Write a synthetic H5 file that mimics Keras weight layout."""
    with h5py.File(path, "w") as hf:
        sd = model.state_dict()
        for k, v in sd.items():
            w = v.numpy()
            # Transpose conv weights back to Keras format for the test
            if w.ndim == 5:
                w = np.transpose(w, (2, 3, 4, 1, 0))  # Cout,Cin,D,H,W → D,H,W,Cin,Cout
            hf.create_dataset(
                k.replace(".", "/") + "/kernel" if w.ndim == 5 else k, data=w
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvertWeights:
    def test_returns_dict(self, tmp_path):
        model = _SimplePT()
        h5_path = str(tmp_path / "weights.h5")
        # Write a minimal H5 that has some datasets
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("dummy", data=np.zeros(4))
        result = convert_weights(h5_path, model)
        assert isinstance(result, dict)

    def test_output_pth_written(self, tmp_path):
        model = _SimplePT()
        h5_path = str(tmp_path / "weights.h5")
        pth_path = str(tmp_path / "weights.pth")
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("dummy", data=np.zeros(4))
        convert_weights(h5_path, model, output_path=pth_path)
        assert Path(pth_path).exists()
        loaded = torch.load(pth_path, map_location="cpu", weights_only=True)
        assert isinstance(loaded, dict)

    def test_state_dict_keys_preserved(self, tmp_path):
        """Model state dict should have same keys before and after conversion."""
        model = _SimplePT()
        original_keys = set(model.state_dict().keys())
        h5_path = str(tmp_path / "weights.h5")
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("dummy", data=np.zeros(4))
        convert_weights(h5_path, model)
        assert set(model.state_dict().keys()) == original_keys
