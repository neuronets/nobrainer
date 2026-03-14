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
# # Getting Started with Nobrainer (PyTorch)
#
# Nobrainer is a deep learning framework for 3D brain image processing built
# on PyTorch and MONAI. This tutorial covers:
#
# 1. Installation
# 2. Loading a segmentation model
# 3. Running inference on a synthetic volume
# 4. Inspecting the output

# %%
# Install nobrainer (uncomment if needed)
# !pip install nobrainer monai

# %%
import numpy as np
import torch

import nobrainer  # noqa: F401
from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## Create a synthetic brain volume
#
# For demonstration, we create a 64^3 volume with a bright sphere
# (simulating a brain structure).


# %%
def make_sphere_volume(shape=(64, 64, 64), radius=20):
    """Synthetic volume with a centered sphere."""
    vol = np.random.rand(*shape).astype(np.float32) * 0.3
    center = np.array(shape) / 2
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    vol[dist < radius] += 0.7
    label = (dist < radius).astype(np.float32)
    return vol, label


vol, label = make_sphere_volume()
print(f"Volume shape: {vol.shape}, Label shape: {label.shape}")
print(f"Foreground voxels: {label.sum():.0f}")

# %% [markdown]
# ## Instantiate and inspect a UNet

# %%
model = unet(n_classes=2)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model.__class__.__name__}")
print(f"Parameters: {n_params:,}")

# Quick forward pass
x = torch.randn(1, 1, 64, 64, 64)
with torch.no_grad():
    out = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")

# %% [markdown]
# ## Run block-based prediction
#
# `predict()` splits the volume into blocks, runs inference, and
# stitches the result back together.

# %%
result = predict(
    inputs=vol,
    model=model,
    block_shape=(32, 32, 32),
    batch_size=4,
    device="cpu",
    return_labels=True,
)

pred = np.asarray(result.dataobj)
print(f"Prediction shape: {pred.shape}")
print(f"Unique labels: {np.unique(pred)}")

# %% [markdown]
# The model is randomly initialized so predictions won't be meaningful.
# To get useful results, load trained weights:
#
# ```python
# model.load_state_dict(torch.load("unet_brainmask.pth"))
# ```
#
# Pre-trained models are available at
# https://github.com/neuronets/trained-models

# %% [markdown]
# ## Available models
#
# Nobrainer provides several model families:

# %%
from nobrainer.models import get as get_model  # noqa: E402

for name in [
    "unet",
    "vnet",
    "attention_unet",
    "unetr",
    "meshnet",
    "highresnet",
    "bayesian_vnet",
    "bayesian_meshnet",
    "progressivegan",
    "dcgan",
]:
    try:
        factory = get_model(name)
        m = factory(n_classes=2) if "gan" not in name else factory(latent_size=8)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:20s}  {n:>10,} params")
    except Exception as e:
        print(f"  {name:20s}  (requires extras: {e})")
