"""SR-test: SynthSeg generation from real aparc+aseg label maps.

Tests that the enhanced SynthSeg generator produces realistic synthetic
images from actual FreeSurfer parcellation data.
"""

from __future__ import annotations

import numpy as np
import torch


class TestSynthSegBrain:
    """SynthSeg with real brain data."""

    def test_generate_from_sample_data(self, sample_data):
        """Generate synthetic image from real aparc+aseg label map."""
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        # sample_data is list of (image, label) tuples
        label_paths = [p[1] for p in sample_data[:2]]

        gen = SynthSegGenerator(
            label_paths,
            n_samples_per_map=2,
            elastic_std=2.0,  # mild deformation for speed
            rotation_range=10.0,
            randomize_resolution=False,  # skip for speed
        )

        sample = gen[0]
        assert sample["image"].shape[0] == 1  # channel dim
        assert sample["label"].shape[0] == 1
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.int64

        # Image should have non-zero values in brain region
        img = sample["image"][0].numpy()
        lbl = sample["label"][0].numpy()
        brain_mask = lbl > 0
        assert brain_mask.sum() > 0
        assert img[brain_mask].std() > 0  # not constant

    def test_two_samples_differ(self, sample_data):
        """Two samples from same label map should differ."""
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        label_paths = [p[1] for p in sample_data[:1]]
        gen = SynthSegGenerator(
            label_paths,
            n_samples_per_map=2,
            elastic_std=0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=False,
        )

        s1 = gen[0]["image"]
        s2 = gen[1]["image"]
        assert not torch.allclose(s1, s2)

    def test_label_structure_preserved(self, sample_data):
        """Spatial augmentation should preserve label topology."""
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        label_paths = [p[1] for p in sample_data[:1]]
        gen = SynthSegGenerator(
            label_paths,
            n_samples_per_map=1,
            elastic_std=2.0,
            rotation_range=5.0,
        )

        sample = gen[0]
        lbl = sample["label"][0].numpy()

        # Should still have brain structure (not all zeros or all one label)
        unique = np.unique(lbl)
        assert len(unique) > 2  # at least background + 2 regions
