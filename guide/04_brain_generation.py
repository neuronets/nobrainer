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
# This tutorial trains a Progressive GAN on synthetic 3D volumes using
# PyTorch Lightning. In practice, replace with real brain MRI data.
#
# We will:
# 1. Create a synthetic dataset
# 2. Build a ProgressiveGAN model
# 3. Train with PyTorch Lightning
# 4. Generate synthetic volumes

# %%
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.models.generative import ProgressiveGAN

# %% [markdown]
# ## Create synthetic training data
#
# We use small 4^3 volumes to keep this tutorial fast on CPU.
# For real brain generation, use 64^3 or 128^3 volumes with a
# multi-resolution schedule like [4, 8, 16, 32, 64].

# %%
torch.manual_seed(42)

SPATIAL = 4
N_SAMPLES = 64
BATCH_SIZE = 8

# Random "brain-like" volumes (in practice, load real NIfTI data)
imgs = torch.randn(N_SAMPLES, 1, SPATIAL, SPATIAL, SPATIAL)
loader = DataLoader(TensorDataset(imgs), batch_size=BATCH_SIZE, shuffle=True)
print(f"Training set: {N_SAMPLES} volumes of shape {SPATIAL}^3")

# %% [markdown]
# ## Build the ProgressiveGAN
#
# ProgressiveGAN trains at increasing resolutions. Here we use a
# single resolution for the demo. For multi-resolution training,
# pass `resolution_schedule=[4, 8, 16, 32, 64]`.

# %%
model = ProgressiveGAN(
    latent_size=64,
    fmap_base=64,
    fmap_max=64,
    resolution_schedule=[SPATIAL],
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
    max_steps=200,
    accelerator="auto",  # uses GPU if available
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
print(f"Generated shape: {gen_np.shape}")  # (4, 1, 8, 8, 8)
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
