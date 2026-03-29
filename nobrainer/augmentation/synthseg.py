"""SynthSeg-style synthetic brain data generator."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.utils.data


class SynthSegGenerator(torch.utils.data.Dataset):
    """SynthSeg-style synthetic brain data generator.

    Generates synthetic brain images from label maps by sampling random
    intensity statistics per label region, then applying Gaussian noise
    and a smooth polynomial bias field.

    Parameters
    ----------
    label_maps : list of str or Path
        Paths to NIfTI label-map files.
    n_samples_per_map : int
        Number of synthetic samples to generate per label map.
    intensity_range : tuple of float
        ``(min, max)`` for clipping the final image.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    bias_field_order : int
        Polynomial order for the smooth bias field.
    """

    def __init__(
        self,
        label_maps: list[str | Path],
        n_samples_per_map: int = 10,
        intensity_range: tuple[float, float] = (0.0, 1.0),
        noise_std: float = 0.1,
        bias_field_order: int = 3,
    ) -> None:
        self.label_maps = [Path(p) for p in label_maps]
        self.n_samples_per_map = n_samples_per_map
        self.intensity_range = intensity_range
        self.noise_std = noise_std
        self.bias_field_order = bias_field_order

    def __len__(self) -> int:
        return len(self.label_maps) * self.n_samples_per_map

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        map_idx = idx // self.n_samples_per_map
        label_path = self.label_maps[map_idx]

        # Load label map
        label_nii = nib.load(label_path)
        label_data = np.asarray(label_nii.dataobj, dtype=np.int32)

        # Get unique labels and sample random intensity per label
        unique_labels = np.unique(label_data)
        rng = np.random.uniform
        label_means = {lab: rng(0.0, 1.0) for lab in unique_labels}
        label_stds = {lab: rng(0.0, 1.0) for lab in unique_labels}

        # Generate image by filling each label region
        image = np.zeros_like(label_data, dtype=np.float32)
        for lab in unique_labels:
            mask = label_data == lab
            n_voxels = int(mask.sum())
            image[mask] = np.random.normal(
                label_means[lab], label_stds[lab], size=n_voxels
            ).astype(np.float32)

        # Convert to tensor
        image = torch.from_numpy(image).float()

        # Add Gaussian noise
        if self.noise_std > 0:
            image = image + self.noise_std * torch.randn_like(image)

        # Add smooth polynomial bias field
        image = self._add_bias_field(image)

        # Clip to intensity range
        image = image.clamp(self.intensity_range[0], self.intensity_range[1])

        # Add channel dimension: [1, D, H, W]
        image = image.unsqueeze(0)
        label_tensor = torch.from_numpy(label_data).long().unsqueeze(0)

        return {"image": image, "label": label_tensor}

    def _add_bias_field(self, image: torch.Tensor) -> torch.Tensor:
        """Generate and apply a smooth polynomial bias field.

        Parameters
        ----------
        image : torch.Tensor
            3-D image tensor of shape ``(D, H, W)``.

        Returns
        -------
        torch.Tensor
            Image multiplied by the bias field.
        """
        D, H, W = image.shape
        order = self.bias_field_order

        # Create normalised coordinate grids in [-1, 1]
        coords_d = torch.linspace(-1, 1, D)
        coords_h = torch.linspace(-1, 1, H)
        coords_w = torch.linspace(-1, 1, W)

        # Generate random polynomial coefficients
        coeffs = torch.randn(order + 1, order + 1, order + 1) * 0.1

        # Evaluate polynomial on 3-D grid
        bias = torch.zeros_like(image)
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    term = coeffs[i, j, k]
                    term = term * coords_d.pow(i).unsqueeze(1).unsqueeze(2)
                    term = term * coords_h.pow(j).unsqueeze(0).unsqueeze(2)
                    term = term * coords_w.pow(k).unsqueeze(0).unsqueeze(1)
                    bias = bias + term

        # Convert to multiplicative field centred around 1
        bias = torch.exp(bias)

        return image * bias
