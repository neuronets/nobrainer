# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Train a Progressive GAN for Brain Generation
#
# This tutorial trains a Progressive GAN on real brain MRI data using
# PyTorch Lightning. The volumes are downsampled to 4^3 resolution for
# fast CPU training.
#
# We will:
# 1. Download real brain MRI data
# 2. Downsample volumes to small resolution
# 3. Build a ProgressiveGAN model
# 4. Train with PyTorch Lightning
# 5. Generate synthetic volumes

# %%
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.io import read_csv
from nobrainer.models.generative import ProgressiveGAN
from nobrainer.utils import get_data

# %% [markdown]
# ## Download and downsample real brain MRI data
#
# We load all 10 T1-weighted volumes, normalize each to [0, 1], and
# downsample to 4^3 for fast GAN training on CPU.

# %%
TARGET_SIZE = 4

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subjects")

volumes = []
for img_path, _ in filepaths:
    vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
    # Normalize to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    # Downsample to TARGET_SIZE^3
    factors = [TARGET_SIZE / s for s in vol.shape[:3]]
    vol_small = zoom(vol, factors, order=1)
    volumes.append(vol_small)

imgs = torch.from_numpy(np.stack(volumes)[:, None])  # (N, 1, 4, 4, 4)
print(f"Training set: {imgs.shape[0]} volumes of shape {TARGET_SIZE}^3")
print(f"Value range: [{imgs.min():.3f}, {imgs.max():.3f}]")

# %%
torch.manual_seed(42)

BATCH_SIZE = 4
loader = DataLoader(TensorDataset(imgs), batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## Build the ProgressiveGAN
#
# We use a single resolution level (4) to keep this demo fast on CPU.
# For multi-resolution training, pass e.g.
# `resolution_schedule=[4, 8, 16, 32, 64]`.

# %%
model = ProgressiveGAN(
    latent_size=64,
    fmap_base=64,
    fmap_max=64,
    resolution_schedule=[TARGET_SIZE],
    steps_per_phase=500,
)

n_params_g = sum(p.numel() for p in model.generator.parameters())
n_params_d = sum(p.numel() for p in model.discriminator.parameters())
print(f"Generator params:     {n_params_g:,}")
print(f"Discriminator params: {n_params_d:,}")

# %% [markdown]
# ## Train with Lightning

# %%
trainer = pl.Trainer(
    max_steps=100,
    accelerator="auto",
    devices=1,
    enable_checkpointing=False,
    logger=False,
    enable_progress_bar=True,
)

trainer.fit(model, loader)
print(f"Training complete: {model._step_count} steps")

# %% [markdown]
# ## Generate synthetic volumes

# %%
model.eval()
model.generator.current_level = 0
model.generator.alpha = 1.0

with torch.no_grad():
    z = torch.randn(4, 64, device=model.device)
    generated = model.generator(z)

gen_np = generated.cpu().numpy()
print(f"Generated shape: {gen_np.shape}")
print(f"Value range: [{gen_np.min():.3f}, {gen_np.max():.3f}]")
print(f"Std: {gen_np.std():.4f}")

# %% [markdown]
# ## Save and load
#
# ```python
# # Save Lightning checkpoint
# trainer.save_checkpoint("progressivegan.ckpt")
#
# # Load and generate
# model = ProgressiveGAN.load_from_checkpoint("progressivegan.ckpt")
# ```
#
# ## CLI generation
#
# ```bash
# nobrainer generate \
#   --model progressivegan.ckpt \
#   --model-type progressivegan \
#   output_brain.nii.gz
# ```
