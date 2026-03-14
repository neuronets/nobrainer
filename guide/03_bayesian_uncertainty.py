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
# # Bayesian Inference with Uncertainty Maps
#
# This tutorial demonstrates how to use Bayesian neural networks in
# Nobrainer for segmentation with uncertainty quantification.
#
# We will:
# 1. Train a Bayesian VNet on synthetic data
# 2. Run Monte-Carlo inference to get uncertainty maps
# 3. Inspect label, variance, and entropy outputs

# %%
import numpy as np
import torch
import torch.nn as nn

from nobrainer.models.bayesian import BayesianVNet
from nobrainer.prediction import predict_with_uncertainty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Create synthetic data

# %%
torch.manual_seed(42)
np.random.seed(42)

SPATIAL = 64


def make_sphere(spatial=64, radius=18):
    vol = np.random.rand(spatial, spatial, spatial).astype(np.float32) * 0.3
    label = np.zeros((spatial, spatial, spatial), dtype=np.int64)
    center = np.array([spatial // 2] * 3)
    coords = np.mgrid[:spatial, :spatial, :spatial]
    dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
    mask = dist < radius
    vol[mask] += 0.7
    label[mask] = 1
    return vol, label


vol, label = make_sphere(SPATIAL)
x = torch.from_numpy(vol[None, None]).to(device)
y = torch.from_numpy(label).long().to(device)

print(f"Volume: {vol.shape}, Label classes: {np.unique(label)}")

# %% [markdown]
# ## Build and train a Bayesian VNet
#
# The Bayesian VNet uses Pyro-based stochastic weight layers. Each
# forward pass samples different weights, giving different predictions.

# %%
model = BayesianVNet(in_channels=1, n_classes=2, prior_type="standard_normal").to(
    device
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y.unsqueeze(0))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch + 1:3d}: loss={loss.item():.4f}")

# %% [markdown]
# ## Monte-Carlo inference with uncertainty
#
# `predict_with_uncertainty()` runs `n_samples` stochastic forward
# passes and computes:
# - **label**: argmax of mean softmax probabilities
# - **variance**: mean predictive variance across classes
# - **entropy**: predictive entropy of the mean distribution
#
# Higher variance/entropy indicates the model is less certain.

# %%
label_img, var_img, entropy_img = predict_with_uncertainty(
    inputs=vol,
    model=model,
    n_samples=10,
    block_shape=(32, 32, 32),
    batch_size=4,
    device=str(device),
)

print(f"Label shape:    {label_img.shape}")
print(f"Variance shape: {var_img.shape}")
print(f"Entropy shape:  {entropy_img.shape}")

# %%
label_arr = np.asarray(label_img.dataobj)
var_arr = np.asarray(var_img.dataobj)
entropy_arr = np.asarray(entropy_img.dataobj)

# Dice score
pred_bin = (label_arr == 1).astype(np.float32)
label_bin = (label == 1).astype(np.float32)
intersection = (pred_bin * label_bin).sum()
dice = 2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)  # noqa: E226

print(f"Dice score: {dice:.4f}")
print(f"Variance range: [{var_arr.min():.6f}, {var_arr.max():.6f}]")
print(f"Entropy range:  [{entropy_arr.min():.6f}, {entropy_arr.max():.6f}]")
print(f"Mean variance:  {var_arr.mean():.6f}")

# %% [markdown]
# ## Interpreting uncertainty
#
# - Voxels with high variance/entropy are where the model is uncertain
# - Typically, uncertainty is highest near class boundaries
# - This is useful for quality control and active learning
#
# To save the outputs:
# ```python
# label_img.to_filename("label.nii.gz")
# var_img.to_filename("variance.nii.gz")
# entropy_img.to_filename("entropy.nii.gz")
# ```
