"""Tests for Bayesian segmentation with uncertainty quantification.

These tests train a BayesianVNet and run MC prediction on real brain
data, which takes 4+ minutes on CPU.  They are marked ``@pytest.mark.gpu``
so they only run on the EC2 GPU runner (where they take <30s).
The same functionality is also covered by ``test_kwyk_smoke.py``.
"""

import nibabel as nib
import numpy as np
import pytest

pyro = pytest.importorskip("pyro")  # noqa: F841

from nobrainer.processing import Dataset, Segmentation  # noqa: E402


@pytest.mark.gpu
class TestBayesianUncertainty:
    """Test Bayesian model produces uncertainty estimates."""

    def test_bayesian_predict_returns_tuple(self, train_eval_split, tmp_path):
        """Bayesian predict with n_samples returns (label, variance, entropy)."""
        train_data, eval_pair = train_eval_split
        eval_img_path = eval_pair[0]

        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
        )

        seg = Segmentation("bayesian_vnet")
        seg.fit(ds, epochs=2)
        result = seg.predict(eval_img_path, block_shape=(16, 16, 16), n_samples=3)

        # Should return a tuple of 3 NIfTI images
        assert isinstance(result, tuple)
        assert len(result) == 3

        label, variance, entropy = result
        assert isinstance(label, nib.Nifti1Image)
        assert isinstance(variance, nib.Nifti1Image)
        assert isinstance(entropy, nib.Nifti1Image)

    def test_variance_nonzero(self, train_eval_split, tmp_path):
        """Bayesian model variance should be non-zero."""
        train_data, eval_pair = train_eval_split
        eval_img_path = eval_pair[0]

        ds = (
            Dataset.from_files(
                train_data,
                block_shape=(16, 16, 16),
                n_classes=2,
            )
            .batch(2)
            .binarize()
        )

        seg = Segmentation("bayesian_vnet")
        seg.fit(ds, epochs=2)
        _, variance, _ = seg.predict(
            eval_img_path, block_shape=(16, 16, 16), n_samples=3
        )

        var_data = np.asarray(variance.dataobj)
        assert np.any(var_data > 0), "Variance should be non-zero"
