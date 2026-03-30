"""Unit tests for class weight computation and weighted losses."""

from __future__ import annotations

import numpy as np
import torch

from nobrainer.losses import DiceCELoss, compute_class_weights, weighted_cross_entropy


class TestComputeClassWeights:
    def test_uniform_distribution(self, tmp_path):
        """Equal class counts → all weights ≈ 1."""
        import nibabel as nib

        # Create 2-class volume with equal counts
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[:5] = 1  # half zeros, half ones
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tmp_path / "lbl.nii.gz"))

        w = compute_class_weights([str(tmp_path / "lbl.nii.gz")], n_classes=2)
        assert w.shape == (2,)
        assert torch.allclose(w, torch.ones(2), atol=0.01)

    def test_imbalanced_gives_higher_weight_to_rare(self, tmp_path):
        """Rare class should get higher weight."""
        import nibabel as nib

        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[0, 0, 0] = 1  # class 1 is very rare
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tmp_path / "lbl.nii.gz"))

        w = compute_class_weights([str(tmp_path / "lbl.nii.gz")], n_classes=2)
        assert w[1] > w[0]  # rare class gets higher weight

    def test_median_frequency_method(self, tmp_path):
        import nibabel as nib

        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[:2] = 1
        arr[:1] = 2
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tmp_path / "lbl.nii.gz"))

        w = compute_class_weights(
            [str(tmp_path / "lbl.nii.gz")],
            n_classes=3,
            method="median_frequency",
        )
        assert w.shape == (3,)
        assert (w > 0).all()

    def test_max_samples(self, tmp_path):
        """max_samples limits the number of files scanned."""
        import nibabel as nib

        for i in range(5):
            arr = np.full((4, 4, 4), i % 2, dtype=np.int32)
            nib.save(
                nib.Nifti1Image(arr, np.eye(4)),
                str(tmp_path / f"lbl_{i}.nii.gz"),
            )

        paths = [str(tmp_path / f"lbl_{i}.nii.gz") for i in range(5)]
        w = compute_class_weights(paths, n_classes=2, max_samples=2)
        assert w.shape == (2,)


class TestWeightedCrossEntropy:
    def test_with_weights(self):
        w = torch.tensor([0.5, 1.5])
        loss_fn = weighted_cross_entropy(weight=w)
        pred = torch.randn(4, 2)
        target = torch.randint(0, 2, (4,))
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_without_weights(self):
        loss_fn = weighted_cross_entropy()
        pred = torch.randn(4, 2)
        target = torch.randint(0, 2, (4,))
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)


class TestDiceCELoss:
    def test_3d_segmentation(self):
        loss_fn = DiceCELoss(softmax=True)
        pred = torch.randn(2, 3, 8, 8, 8)  # 3-class
        target = torch.randint(0, 3, (2, 8, 8, 8))
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_with_class_weights(self):
        w = torch.tensor([0.5, 1.0, 2.0])
        loss_fn = DiceCELoss(weight=w, softmax=True)
        pred = torch.randn(2, 3, 8, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8, 8))
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_loss_registry(self):
        from nobrainer.losses import get

        loss_cls = get("dice_ce")
        assert loss_cls is DiceCELoss
