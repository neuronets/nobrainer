"""Tests for extract_patches() with various binarization modes."""

import nibabel as nib
import numpy as np
import pytest

from nobrainer.processing import extract_patches


class TestExtractPatches:
    """Test extract_patches() on real brain data."""

    @pytest.fixture()
    def volume_and_label(self, train_eval_split):
        """Load first volume and label as numpy arrays."""
        train_data, _ = train_eval_split
        img_path, lbl_path = train_data[0]
        vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        lbl = np.asarray(nib.load(lbl_path).dataobj, dtype=np.float32)
        return vol, lbl

    def test_binarize_true(self, volume_and_label):
        """binarize=True maps any non-zero label to 1."""
        vol, lbl = volume_and_label
        patches = extract_patches(
            vol, lbl, block_shape=(16, 16, 16), n_patches=5, binarize=True
        )
        assert len(patches) == 5
        for img_patch, lbl_patch in patches:
            assert img_patch.shape == (16, 16, 16)
            assert lbl_patch.shape == (16, 16, 16)
            # Only 0 and 1 in binarized labels
            unique_vals = set(np.unique(lbl_patch))
            assert unique_vals <= {0.0, 1.0}

    def test_binarize_set(self, volume_and_label):
        """binarize={17, 53} selects hippocampus labels only."""
        vol, lbl = volume_and_label
        patches = extract_patches(
            vol, lbl, block_shape=(16, 16, 16), n_patches=5, binarize={17, 53}
        )
        for img_patch, lbl_patch in patches:
            assert img_patch.shape == (16, 16, 16)
            unique_vals = set(np.unique(lbl_patch))
            assert unique_vals <= {0.0, 1.0}

    def test_binarize_callable(self, volume_and_label):
        """binarize=lambda applies custom function to label patches."""
        vol, lbl = volume_and_label

        def threshold_fn(x):
            return (x >= 1000).astype(np.float32)

        patches = extract_patches(
            vol, lbl, block_shape=(16, 16, 16), n_patches=5, binarize=threshold_fn
        )
        for img_patch, lbl_patch in patches:
            assert img_patch.shape == (16, 16, 16)
            unique_vals = set(np.unique(lbl_patch))
            assert unique_vals <= {0.0, 1.0}

    def test_patch_shapes(self, volume_and_label):
        """Patches have the requested block_shape."""
        vol, lbl = volume_and_label
        patches = extract_patches(vol, lbl, block_shape=(16, 16, 16), n_patches=3)
        assert len(patches) == 3
        for img_patch, lbl_patch in patches:
            assert img_patch.shape == (16, 16, 16)
            assert lbl_patch.shape == (16, 16, 16)
