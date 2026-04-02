"""SynthSeg-style synthetic brain data generator.

Enhanced implementation following Billot et al. (2023) with:
- GMM tissue class grouping (labels grouped by tissue type)
- Spatial augmentation (elastic deformation, rotation, scaling, flipping)
- Resolution randomization (downsample + upsample)
- Configurable intensity priors

Reference: Billot et al., "SynthSeg: Segmentation of brain MRI scans
of any contrast and resolution without retraining", Medical Image Analysis, 2023.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.utils.data


class SynthSegGenerator(torch.utils.data.Dataset):
    """SynthSeg-style synthetic brain data generator.

    Generates synthetic brain images from label maps with domain
    randomization for contrast-agnostic training.

    Parameters
    ----------
    label_maps : list of str or Path
        Paths to NIfTI label-map files (e.g., FreeSurfer aparc+aseg).
    n_samples_per_map : int
        Number of synthetic samples per label map.
    generation_classes : dict or None
        Tissue class grouping: ``{"WM": [2, 41], ...}``.
        Labels in the same class share one intensity distribution.
        None = use default FreeSurfer tissue classes.
    intensity_prior : tuple of float
        ``(min, max)`` bounds for sampling per-class mean intensities.
    std_prior : tuple of float
        ``(min, max)`` bounds for sampling per-class std.
    noise_std : float
        Additive Gaussian noise std.
    bias_field_std : float
        Bias field magnitude (std of polynomial coefficients).
    elastic_std : float
        Elastic deformation magnitude (0 = disabled).
    rotation_range : float
        Max rotation in degrees per axis (0 = disabled).
    scaling_bounds : float
        Max scaling fraction (e.g., 0.2 = ±20%).
    flipping : bool
        Enable random left-right flipping with label remapping.
    randomize_resolution : bool
        Simulate variable acquisition resolution.
    resolution_range : tuple of float
        ``(min_mm, max_mm)`` per-axis resolution range.
    """

    def __init__(
        self,
        label_maps: list[str | Path] | None = None,
        n_samples_per_map: int = 10,
        generation_classes: dict[str, list[int]] | None = None,
        intensity_prior: tuple[float, float] = (0.0, 250.0),
        std_prior: tuple[float, float] = (0.0, 35.0),
        noise_std: float = 0.1,
        bias_field_std: float = 0.7,
        elastic_std: float = 4.0,
        rotation_range: float = 15.0,
        scaling_bounds: float = 0.2,
        flipping: bool = True,
        randomize_resolution: bool = True,
        resolution_range: tuple[float, float] = (1.0, 3.0),
        seed: int | None = None,
        zarr_store: str | Path | None = None,
        zarr_level: int = 0,
    ) -> None:
        if label_maps is not None:
            self.label_maps = [Path(p) for p in label_maps]
        else:
            self.label_maps = []
        self._seed = seed

        # Zarr store support: read labels from a pyramidal zarr store
        self._zarr_store_path = str(zarr_store) if zarr_store else None
        self._zarr_level = zarr_level
        self._zarr_group = None
        self._zarr_n_subjects = 0
        if self._zarr_store_path is not None:
            import zarr

            self._zarr_group = zarr.open_group(self._zarr_store_path, mode="r")
            lbl_arr = self._zarr_group[f"labels/{zarr_level}"]
            self._zarr_n_subjects = lbl_arr.shape[0]
            if not self.label_maps:
                # Use zarr subjects as the "label maps"
                self.label_maps = list(range(self._zarr_n_subjects))
        self.n_samples_per_map = n_samples_per_map
        self.intensity_prior = intensity_prior
        self.std_prior = std_prior
        self.noise_std = noise_std
        self.bias_field_std = bias_field_std
        self.elastic_std = elastic_std
        self.rotation_range = rotation_range
        self.scaling_bounds = scaling_bounds
        self.flipping = flipping
        self.randomize_resolution = randomize_resolution
        self.resolution_range = resolution_range

        # Load tissue class mapping
        if generation_classes is None:
            from nobrainer.data.tissue_classes import FREESURFER_TISSUE_CLASSES

            self.generation_classes = FREESURFER_TISSUE_CLASSES
        else:
            self.generation_classes = generation_classes

        # Build reverse lookup: label_id → class_name
        self._label_to_class: dict[int, str] = {}
        for cls_name, label_ids in self.generation_classes.items():
            for lid in label_ids:
                self._label_to_class[lid] = cls_name

    def __len__(self) -> int:
        n_maps = self._zarr_n_subjects if self._zarr_group else len(self.label_maps)
        return n_maps * self.n_samples_per_map

    def _get_rng(self, idx: int) -> np.random.Generator:
        """Get a seeded RNG for reproducibility, or unseeded if no seed."""
        if self._seed is not None:
            return np.random.default_rng(self._seed + idx)
        return np.random.default_rng()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        map_idx = idx // self.n_samples_per_map

        # Load label map from zarr store or NIfTI
        if self._zarr_group is not None:
            subject_idx = map_idx % self._zarr_n_subjects
            lbl_arr = self._zarr_group[f"labels/{self._zarr_level}"]
            label_data = np.asarray(lbl_arr[subject_idx], dtype=np.int32)
        else:
            label_path = self.label_maps[map_idx]
            label_data = np.asarray(nib.load(label_path).dataobj, dtype=np.int32)

        # 1. GMM intensity generation (per tissue class)
        image = self._generate_intensities(label_data)

        # 2. Spatial augmentation (elastic + affine + flip)
        if self.elastic_std > 0 or self.rotation_range > 0 or self.flipping:
            image, label_data = self._spatial_augmentation(image, label_data)

        # 3. Resolution randomization
        if self.randomize_resolution:
            image = self._randomize_resolution(image)

        # 4. Bias field
        if self.bias_field_std > 0:
            image = self._add_bias_field(image)

        # 5. Gaussian noise
        if self.noise_std > 0:
            image = image + np.random.normal(0, self.noise_std, image.shape).astype(
                np.float32
            )

        # Convert to tensors with channel dim [1, D, H, W]
        image_t = torch.from_numpy(image).float().unsqueeze(0)
        label_t = torch.from_numpy(label_data).long().unsqueeze(0)

        return {"image": image_t, "label": label_t}

    # ------------------------------------------------------------------
    # GMM intensity generation
    # ------------------------------------------------------------------

    def _generate_intensities(self, label_data: np.ndarray) -> np.ndarray:
        """Generate image by sampling GMM intensities per tissue class."""
        rng = np.random.default_rng()
        unique_labels = np.unique(label_data)

        # Sample one (mean, std) per tissue class
        class_params: dict[str, tuple[float, float]] = {}
        for cls_name in self.generation_classes:
            mean = rng.uniform(*self.intensity_prior)
            std = rng.uniform(*self.std_prior)
            class_params[cls_name] = (mean, std)

        # Fill each label region from its class distribution
        image = np.zeros_like(label_data, dtype=np.float32)
        for lab in unique_labels:
            mask = label_data == lab
            n_vox = int(mask.sum())
            if n_vox == 0:
                continue

            cls_name = self._label_to_class.get(lab)
            if cls_name is not None and cls_name in class_params:
                mean, std = class_params[cls_name]
            else:
                # Unknown label: sample fresh random params
                mean = rng.uniform(*self.intensity_prior)
                std = rng.uniform(*self.std_prior)

            image[mask] = rng.normal(mean, max(std, 1e-6), size=n_vox).astype(
                np.float32
            )

        return image

    # ------------------------------------------------------------------
    # Spatial augmentation
    # ------------------------------------------------------------------

    def _spatial_augmentation(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation, affine transform, and flipping."""
        from scipy.ndimage import map_coordinates

        D, H, W = image.shape

        # Build coordinate grid
        coords = np.mgrid[:D, :H, :W].astype(np.float32)  # (3, D, H, W)

        # Elastic deformation: smooth random displacement field
        if self.elastic_std > 0:
            # Sample on coarse grid, smooth, then resize
            coarse_shape = (max(4, D // 8), max(4, H // 8), max(4, W // 8))
            rng = np.random.default_rng()
            for axis in range(3):
                displacement = rng.normal(0, self.elastic_std, coarse_shape).astype(
                    np.float32
                )
                # Smooth
                from scipy.ndimage import gaussian_filter, zoom

                displacement = gaussian_filter(displacement, sigma=2.0)
                # Resize to full volume
                zoom_factors = (
                    D / coarse_shape[0],
                    H / coarse_shape[1],
                    W / coarse_shape[2],
                )
                displacement = zoom(displacement, zoom_factors, order=1)
                # Crop/pad to exact shape if needed
                displacement = displacement[:D, :H, :W]
                coords[axis] += displacement

        # Affine: rotation + scaling
        if self.rotation_range > 0 or self.scaling_bounds > 0:
            center = np.array([D / 2, H / 2, W / 2])
            coords_centered = coords.reshape(3, -1) - center[:, None]

            # Build rotation matrix (Euler angles)
            rng = np.random.default_rng()
            angles = rng.uniform(-self.rotation_range, self.rotation_range, size=3)
            angles_rad = np.deg2rad(angles)
            Rx = _rot_x(angles_rad[0])
            Ry = _rot_y(angles_rad[1])
            Rz = _rot_z(angles_rad[2])
            R = Rz @ Ry @ Rx

            # Scaling
            if self.scaling_bounds > 0:
                scale = rng.uniform(
                    1 - self.scaling_bounds, 1 + self.scaling_bounds, size=3
                )
                S = np.diag(scale)
                R = R @ S

            coords_centered = R @ coords_centered
            coords = (coords_centered + center[:, None]).reshape(3, D, H, W)

        # Apply spatial transform
        image_out = map_coordinates(image, coords, order=3, mode="nearest")
        label_out = map_coordinates(
            label.astype(np.float32), coords, order=0, mode="nearest"
        ).astype(np.int32)

        # Flipping
        if self.flipping and np.random.random() > 0.5:
            image_out = np.flip(image_out, axis=2).copy()  # flip W axis (L/R)
            label_out = np.flip(label_out, axis=2).copy()
            label_out = self._remap_lr_labels(label_out)

        return image_out.astype(np.float32), label_out

    @staticmethod
    def _remap_lr_labels(label: np.ndarray) -> np.ndarray:
        """Swap left/right FreeSurfer labels after L/R flip."""
        from nobrainer.data.tissue_classes import FREESURFER_LR_PAIRS

        result = label.copy()
        for left, right in FREESURFER_LR_PAIRS:
            left_mask = label == left
            right_mask = label == right
            result[left_mask] = right
            result[right_mask] = left
        return result

    # ------------------------------------------------------------------
    # Resolution randomization
    # ------------------------------------------------------------------

    def _randomize_resolution(self, image: np.ndarray) -> np.ndarray:
        """Simulate variable MRI acquisition resolution."""
        from scipy.ndimage import gaussian_filter, zoom

        rng = np.random.default_rng()
        target_res = rng.uniform(*self.resolution_range, size=3)

        # Downsample with anti-aliasing
        sigmas = [max(0, (r - 1) / 2) for r in target_res]
        blurred = gaussian_filter(image, sigma=sigmas)

        # Downsample then upsample
        down_factors = [1.0 / r for r in target_res]
        downsampled = zoom(blurred, down_factors, order=1)
        up_factors = [image.shape[i] / downsampled.shape[i] for i in range(3)]
        upsampled = zoom(downsampled, up_factors, order=1)

        # Ensure exact shape match
        D, H, W = image.shape
        return upsampled[:D, :H, :W].astype(np.float32)

    # ------------------------------------------------------------------
    # Bias field
    # ------------------------------------------------------------------

    def _add_bias_field(self, image: np.ndarray) -> np.ndarray:
        """Apply smooth multiplicative bias field."""
        D, H, W = image.shape
        order = 3

        coords_d = np.linspace(-1, 1, D)
        coords_h = np.linspace(-1, 1, H)
        coords_w = np.linspace(-1, 1, W)

        rng = np.random.default_rng()
        coeffs = rng.normal(0, self.bias_field_std, (order + 1, order + 1, order + 1))

        bias = np.zeros_like(image)
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    term = coeffs[i, j, k]
                    term = term * np.power(coords_d, i)[:, None, None]
                    term = term * np.power(coords_h, j)[None, :, None]
                    term = term * np.power(coords_w, k)[None, None, :]
                    bias += term

        bias = np.exp(bias)
        return (image * bias).astype(np.float32)


# ------------------------------------------------------------------
# Rotation matrix helpers
# ------------------------------------------------------------------


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
