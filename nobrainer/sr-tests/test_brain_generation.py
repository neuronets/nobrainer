"""Tests for brain generation with Progressive GAN."""

import nibabel as nib
import pytest

pl = pytest.importorskip("pytorch_lightning")  # noqa: F841

from nobrainer.processing import Dataset, Generation  # noqa: E402


class TestBrainGeneration:
    """Test generative model training and image generation."""

    def test_generate_returns_nifti_images(self, train_eval_split, tmp_path):
        """Generation.fit().generate(2) returns 2 NIfTI images."""
        train_data, _ = train_eval_split

        ds = Dataset.from_files(
            train_data,
            block_shape=(16, 16, 16),
            n_classes=1,
        ).batch(2)

        gen = Generation("progressivegan")
        gen.fit(ds, epochs=50)
        images = gen.generate(n_images=2)

        assert len(images) == 2
        for img in images:
            assert isinstance(img, nib.Nifti1Image)
            assert len(img.shape) >= 3
