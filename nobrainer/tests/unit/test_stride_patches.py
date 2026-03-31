"""Unit tests for strided patch extraction and reassembly."""

from __future__ import annotations

import numpy as np

from nobrainer.prediction import reassemble_predictions, strided_patch_positions


class TestStridedPatchPositions:
    def test_non_overlapping_count(self):
        """256³ with block=32 stride=32 → 8³ = 512 patches."""
        pos = strided_patch_positions((256, 256, 256), (32, 32, 32), (32, 32, 32))
        assert len(pos) == 8 * 8 * 8  # 512

    def test_overlapping_more_patches(self):
        """Stride < block produces more patches."""
        non_overlap = strided_patch_positions((64, 64, 64), (32, 32, 32), (32, 32, 32))
        overlap = strided_patch_positions((64, 64, 64), (32, 32, 32), (16, 16, 16))
        assert len(overlap) > len(non_overlap)

    def test_patch_shapes_valid(self):
        """Each position should yield a valid slice."""
        pos = strided_patch_positions((100, 100, 100), (32, 32, 32), (16, 16, 16))
        for sd, sh, sw in pos:
            assert sd.stop - sd.start == 32
            assert sh.stop - sh.start == 32
            assert sw.stop - sw.start == 32
            assert sd.stop <= 100
            assert sh.stop <= 100
            assert sw.stop <= 100

    def test_stride_equals_block_default(self):
        """None stride defaults to block_shape."""
        pos = strided_patch_positions((64, 64, 64), (32, 32, 32))
        assert len(pos) == 2 * 2 * 2  # 8


class TestReassemblePredictions:
    def test_non_overlapping_perfect_reconstruction(self):
        """Non-overlapping patches reassemble perfectly."""
        vol_shape = (64, 64, 64)
        block = (32, 32, 32)
        n_classes = 2

        # Create a known volume
        original = np.random.randn(n_classes, *vol_shape).astype(np.float32)

        # Extract non-overlapping patches
        positions = strided_patch_positions(vol_shape, block, block)
        patches = []
        for sd, sh, sw in positions:
            patch = original[:, sd, sh, sw]
            patches.append((patch, (sd, sh, sw)))

        # Reassemble
        result = reassemble_predictions(patches, vol_shape, n_classes)
        assert np.allclose(result, original, atol=1e-6)

    def test_overlapping_average(self):
        """Overlapping patches with averaging should still reconstruct reasonably."""
        vol_shape = (64, 64, 64)
        block = (32, 32, 32)
        stride = (16, 16, 16)
        n_classes = 2

        # Create constant volume (averaging constant = constant)
        original = np.ones((n_classes, *vol_shape), dtype=np.float32) * 0.5

        positions = strided_patch_positions(vol_shape, block, stride)
        patches = []
        for sd, sh, sw in positions:
            patch = original[:, sd, sh, sw]
            patches.append((patch, (sd, sh, sw)))

        result = reassemble_predictions(
            patches, vol_shape, n_classes, strategy="average"
        )
        assert np.allclose(result, 0.5, atol=1e-5)

    def test_output_shape(self):
        """Output shape matches volume_shape."""
        patches = [
            (np.ones((3, 16, 16, 16)), (slice(0, 16), slice(0, 16), slice(0, 16)))
        ]
        result = reassemble_predictions(patches, (32, 32, 32), 3)
        assert result.shape == (3, 32, 32, 32)
