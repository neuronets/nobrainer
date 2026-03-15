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
# Nobrainer for segmentation with uncertainty quantification, using
# real brain MRI data.
#
# We will:
# 1. Load a real T1-weighted brain volume
# 2. Train a small Bayesian VNet on random patches
# 3. Run Monte-Carlo inference to get uncertainty maps
# 4. Inspect label, variance, and entropy outputs

# %%
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from nobrainer.io import read_csv
from nobrainer.models.bayesian import BayesianVNet
from nobrainer.prediction import predict_with_uncertainty
from nobrainer.utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Load real brain MRI data

# %%
torch.manual_seed(42)
np.random.seed(42)

csv_path = get_data()
filepaths = read_csv(csv_path)
img_path, label_path = filepaths[0]

img = nib.load(img_path)
vol = np.asarray(img.dataobj, dtype=np.float32)
label = np.asarray(nib.load(label_path).dataobj, dtype=np.int64)
# Binarize labels for brain mask task
label = (label > 0).astype(np.int64)

print(f"Volume shape: {vol.shape}")
print(f"Label classes: {np.unique(label)}")

# %% [markdown]
# ## Extract training patches
#
# We extract a few small random patches from the volume for a brief
# training demonstration.

# %%
BLOCK = 32
N_PATCHES = 8

patches_x, patches_y = [], []
for _ in range(N_PATCHES):
    starts = [np.random.randint(0, max(s - BLOCK, 1)) for s in vol.shape[:3]]
    sl = tuple(slice(s, s + BLOCK) for s in starts)
    patches_x.append(vol[sl])
    patches_y.append(label[sl])

x_train = torch.from_numpy(np.stack(patches_x)[:, None]).to(device)
y_train = torch.from_numpy(np.stack(patches_y)).long().to(device)
print(f"Training patches: x={x_train.shape}, y={y_train.shape}")

# %% [markdown]
# ## Build and train a Bayesian VNet
#
# The Bayesian VNet uses Pyro-based stochastic weight layers. Each
# forward pass samples different weights, giving different predictions.
# We use a small model (2 levels, 8 base filters) for CPU speed.

# %%
model = BayesianVNet(
    in_channels=1,
    n_classes=2,
    base_filters=8,
    levels=2,
    prior_type="standard_normal",
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

BATCH_SIZE = 4
model.train()
for epoch in range(30):
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, N_PATCHES, BATCH_SIZE):
        xb = x_train[i : i + BATCH_SIZE]
        yb = y_train[i : i + BATCH_SIZE]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        avg = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1:3d}: loss={avg:.4f}")

# %% [markdown]
# ## Monte-Carlo inference with uncertainty
#
# `predict_with_uncertainty()` runs `n_samples` stochastic forward
# passes and computes:
# - **label**: argmax of mean softmax probabilities
# - **variance**: mean predictive variance across classes
# - **entropy**: predictive entropy of the mean distribution
#
# We run on the first subject's full volume. Higher variance/entropy
# indicates the model is less certain.

# %%
label_img, var_img, entropy_img = predict_with_uncertainty(
    inputs=img_path,
    model=model,
    n_samples=5,
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

# Dice score against binarized ground truth
gt_label = np.asarray(nib.load(label_path).dataobj, dtype=np.float32)
gt_bin = (gt_label > 0).astype(np.float32)
pred_bin = (label_arr == 1).astype(np.float32)
intersection = (pred_bin * gt_bin).sum()
dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)  # noqa: E226

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
