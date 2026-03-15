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
# 1. Downloading sample brain MRI data
# 2. Loading a T1-weighted volume
# 3. Instantiating a segmentation model
# 4. Running block-based inference
# 5. Listing available model families

# %%
# Install nobrainer (uncomment if needed)
# !pip install nobrainer monai

# %%
import nibabel as nib
import numpy as np
import torch

from nobrainer.io import read_csv
from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict
from nobrainer.utils import get_data

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## Download sample brain MRI data
#
# `get_data()` downloads 10 T1-weighted / FreeSurfer label pairs (~46 MB)
# and returns a CSV path. We parse it with `read_csv()`.

# %%
csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Number of subjects: {len(filepaths)}")
print(f"First pair: {filepaths[0]}")

# %% [markdown]
# ## Load and inspect a T1-weighted volume

# %%
t1_path, label_path = filepaths[0]
img = nib.load(t1_path)
vol = np.asarray(img.dataobj, dtype=np.float32)

print(f"Volume shape: {vol.shape}")
print(f"Voxel range:  [{vol.min():.1f}, {vol.max():.1f}]")
print(f"Affine:\n{img.affine}")

# %% [markdown]
# ## Instantiate and inspect a UNet
#
# We create a small UNet (fewer channels) so this runs fast on CPU.

# %%
model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2))
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model.__class__.__name__}")
print(f"Parameters: {n_params:,}")

# Quick forward pass with a small block
x = torch.randn(1, 1, 32, 32, 32)
with torch.no_grad():
    out = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")

# %% [markdown]
# ## Run block-based prediction on a real brain volume
#
# `predict()` splits the volume into blocks, runs inference, and
# stitches the result back together. The model is randomly initialized
# so predictions will not be meaningful.

# %%
result = predict(
    inputs=t1_path,
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
