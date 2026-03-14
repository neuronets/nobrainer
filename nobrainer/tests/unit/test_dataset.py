"""Unit tests for nobrainer.dataset.get_dataset()."""

from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import pytest

from nobrainer.dataset import get_dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nifti(shape=(16, 16, 16), tmpdir: Path | None = None) -> str:
    """Write a synthetic NIfTI file and return its path."""
    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp())
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmpdir / f"vol_{np.random.randint(0, 1e6)}.nii.gz"
    nib.save(img, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetDataset:
    def test_batch_shape_image_only(self, tmp_path):
        """Verify batch shape (B, 1, D, H, W) for image-only dataset."""
        paths = [_make_nifti((16, 16, 16), tmp_path) for _ in range(3)]
        loader = get_dataset(
            image_paths=paths,
            batch_size=2,
            num_workers=0,
            cache_rate=0.0,
        )
        batch = next(iter(loader))
        assert "image" in batch
        assert batch["image"].ndim == 5  # (B, C, D, H, W)
        assert batch["image"].shape[0] == 2  # batch size
        assert batch["image"].shape[1] == 1  # channel

    def test_batch_shape_with_labels(self, tmp_path):
        """Verify both image and label tensors are returned."""
        image_paths = [_make_nifti((16, 16, 16), tmp_path) for _ in range(2)]
        label_paths = [_make_nifti((16, 16, 16), tmp_path) for _ in range(2)]
        loader = get_dataset(
            image_paths=image_paths,
            label_paths=label_paths,
            batch_size=2,
            num_workers=0,
            cache_rate=0.0,
        )
        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch

    def test_mismatch_raises(self, tmp_path):
        """Mismatched image/label list lengths should raise ValueError."""
        paths = [_make_nifti((16, 16, 16), tmp_path) for _ in range(2)]
        with pytest.raises(ValueError, match="len"):
            get_dataset(
                image_paths=paths,
                label_paths=paths[:1],
                batch_size=1,
                num_workers=0,
            )

    def test_augment_flag(self, tmp_path):
        """augment=True should not crash the dataloader."""
        paths = [_make_nifti((16, 16, 16), tmp_path) for _ in range(2)]
        loader = get_dataset(
            image_paths=paths,
            batch_size=2,
            num_workers=0,
            augment=True,
            cache_rate=0.0,
        )
        batch = next(iter(loader))
        assert batch["image"].shape[1] == 1

    def test_returns_dataloader(self, tmp_path):
        paths = [_make_nifti((16, 16, 16), tmp_path)]
        loader = get_dataset(
            image_paths=paths, batch_size=1, num_workers=0, cache_rate=0.0
        )
        from torch.utils.data import DataLoader

        assert isinstance(loader, DataLoader)
