"""Unit tests for enhanced SynthSeg generator."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


def _make_label_map(tmp_path: Path, shape=(32, 32, 32)) -> str:
    """Create a simple label map with a few regions."""
    arr = np.zeros(shape, dtype=np.int32)
    # Background = 0, WM = 2, GM = 3, CSF = 4, hippocampus L = 17, R = 53
    arr[4:28, 4:28, 4:28] = 2  # WM core
    arr[6:26, 6:26, 6:26] = 3  # GM shell
    arr[12:20, 12:20, 12:20] = 4  # CSF center
    arr[8:12, 8:12, 8:16] = 17  # L hippocampus
    arr[8:12, 8:12, 16:24] = 53  # R hippocampus
    path = str(tmp_path / "label.nii.gz")
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
    return path


class TestTissueClasses:
    def test_all_50class_labels_covered(self):
        from nobrainer.data.tissue_classes import FREESURFER_TISSUE_CLASSES

        all_ids = set()
        for ids in FREESURFER_TISSUE_CLASSES.values():
            all_ids.update(ids)
        # Should cover background + major structures
        assert 0 in all_ids  # background
        assert 2 in all_ids  # L cerebral WM
        assert 41 in all_ids  # R cerebral WM
        assert 17 in all_ids  # L hippocampus
        assert 53 in all_ids  # R hippocampus

    def test_no_label_in_multiple_classes(self):
        from nobrainer.data.tissue_classes import FREESURFER_TISSUE_CLASSES

        seen = {}
        for cls_name, ids in FREESURFER_TISSUE_CLASSES.items():
            for lid in ids:
                assert (
                    lid not in seen
                ), f"Label {lid} in both '{seen[lid]}' and '{cls_name}'"
                seen[lid] = cls_name


class TestGMMGrouping:
    def test_within_class_same_distribution(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator(
            [path],
            n_samples_per_map=1,
            elastic_std=0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=False,
            noise_std=0,
            bias_field_std=0,
        )
        sample = gen[0]
        image = sample["image"][0].numpy()  # (D, H, W)
        label = sample["label"][0].numpy()

        # L hippocampus (17) and R hippocampus (53) are both in "hippocampus" class
        # They should have similar mean intensities (same GMM class)
        l_hip = image[label == 17]
        r_hip = image[label == 53]
        if len(l_hip) > 0 and len(r_hip) > 0:
            # Both drawn from same distribution — means should be close
            mean_diff = abs(l_hip.mean() - r_hip.mean())
            pooled_std = max(l_hip.std(), r_hip.std(), 1e-6)
            cv = mean_diff / pooled_std
            assert cv < 0.5  # within-class similarity

    def test_different_classes_differ(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator(
            [path],
            n_samples_per_map=1,
            elastic_std=0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=False,
            noise_std=0,
            bias_field_std=0,
        )
        sample = gen[0]
        image = sample["image"][0].numpy()
        label = sample["label"][0].numpy()

        wm = image[label == 2]
        csf = image[label == 4]
        # WM and CSF should have different distributions (different classes)
        if len(wm) > 10 and len(csf) > 10:
            # Not guaranteed to differ every time but very likely
            assert wm.mean() != pytest.approx(csf.mean(), abs=1.0)

    def test_two_runs_produce_different_intensities(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator(
            [path],
            n_samples_per_map=2,
            elastic_std=0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=False,
        )
        s1 = gen[0]["image"]
        s2 = gen[1]["image"]
        assert not torch.allclose(s1, s2)


class TestSpatialAugmentation:
    def test_elastic_changes_geometry(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator(
            [path],
            n_samples_per_map=1,
            elastic_std=4.0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=False,
            noise_std=0,
            bias_field_std=0,
        )
        sample = gen[0]
        label = sample["label"][0].numpy()

        # Load original label for comparison
        orig = np.asarray(nib.load(path).dataobj, dtype=np.int32)

        # Elastic deformation should change some voxel positions
        changed = (label != orig).sum()
        total = orig.size
        assert changed / total > 0.01  # at least 1% changed

    def test_label_nearest_neighbor(self, tmp_path):
        """Labels should remain integer-valued after spatial augmentation."""
        from nobrainer.augmentation.synthseg import SynthSegGenerator
        from nobrainer.data.tissue_classes import FREESURFER_LR_PAIRS

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator(
            [path],
            n_samples_per_map=1,
            elastic_std=4.0,
            rotation_range=15.0,
            randomize_resolution=False,
        )
        sample = gen[0]
        label = sample["label"][0].numpy()

        # All values should be valid integers (no interpolation artifacts)
        # Include L/R swapped labels since flipping may have occurred
        orig_labels = set(np.asarray(nib.load(path).dataobj, dtype=np.int32).flat)
        valid_labels = set(orig_labels)
        for left, right in FREESURFER_LR_PAIRS:
            if left in orig_labels:
                valid_labels.add(right)
            if right in orig_labels:
                valid_labels.add(left)
        actual_labels = set(label.flat)
        assert actual_labels.issubset(valid_labels)

    def test_flipping_swaps_lr(self, tmp_path):
        """Flipping should swap L/R FreeSurfer codes."""
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator(
            [path],
            n_samples_per_map=30,
            elastic_std=0,
            rotation_range=0,
            flipping=True,
            randomize_resolution=False,
            noise_std=0,
            bias_field_std=0,
        )
        # After L/R flip, label 17 (L hippocampus) should become 53 and vice versa
        # Check that at least one sample has the swap in the label set
        found_swap = False
        orig = np.asarray(nib.load(path).dataobj, dtype=np.int32)
        for i in range(30):
            label = gen[i]["label"][0].numpy()
            # A flip swaps L/R labels AND mirrors spatially.
            # If the spatial distribution of label 17 differs from original, flip happened
            orig_17_count = (orig == 17).sum()
            new_17_count = (label == 17).sum()
            if orig_17_count > 0 and new_17_count != orig_17_count:
                found_swap = True
                break
        assert found_swap


class TestResolutionRandomization:
    def test_blurs_image(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen_sharp = SynthSegGenerator(
            [path],
            n_samples_per_map=1,
            elastic_std=0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=False,
            noise_std=0,
            bias_field_std=0,
        )
        gen_blur = SynthSegGenerator(
            [path],
            n_samples_per_map=1,
            elastic_std=0,
            rotation_range=0,
            flipping=False,
            randomize_resolution=True,
            resolution_range=(2.0, 3.0),  # force heavy blur
            noise_std=0,
            bias_field_std=0,
        )
        # Use same seed for intensity but different resolution
        np.random.seed(42)
        sharp = gen_sharp[0]["image"][0].numpy()
        np.random.seed(42)
        blurred = gen_blur[0]["image"][0].numpy()

        # Blurred should have less high-frequency energy
        sharp_grad = np.abs(np.diff(sharp, axis=0)).mean()
        blur_grad = np.abs(np.diff(blurred, axis=0)).mean()
        assert blur_grad < sharp_grad


class TestOutputFormat:
    def test_returns_dict_with_correct_keys(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator([path], n_samples_per_map=1)
        sample = gen[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape[0] == 1  # channel dim
        assert sample["label"].shape[0] == 1
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.int64

    def test_correct_length(self, tmp_path):
        from nobrainer.augmentation.synthseg import SynthSegGenerator

        path = _make_label_map(tmp_path)
        gen = SynthSegGenerator([path, path], n_samples_per_map=5)
        assert len(gen) == 10
