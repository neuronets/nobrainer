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
# # Train a Segmentation Model
#
# This tutorial shows how to train a UNet for binary brain segmentation
# using synthetic data. In practice, replace the synthetic data with
# real NIfTI files via `nobrainer.dataset.get_dataset()`.
#
# We will:
# 1. Create synthetic training data
# 2. Build a UNet model
# 3. Train with Dice loss
# 4. Evaluate with the predict() API
# 5. Compute Dice score

# %%
import numpy as np
import torch
import torch.nn as nn

from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Create synthetic training data
#
# We generate a batch of volumes, each containing a sphere at a
# slightly randomized position.

# %%
torch.manual_seed(42)
np.random.seed(42)

SPATIAL = 64
N_SAMPLES = 4
BATCH_SIZE = 2


def make_batch(n, spatial=64):
    """Generate n synthetic (volume, label) pairs."""
    vols, labels = [], []
    for _ in range(n):
        vol = np.random.rand(spatial, spatial, spatial).astype(np.float32) * 0.3
        label = np.zeros((spatial, spatial, spatial), dtype=np.int64)
        center = np.array([spatial // 2, spatial // 2, spatial // 2])
        center += np.random.randint(-5, 6, size=3)
        coords = np.mgrid[:spatial, :spatial, :spatial]
        dist = np.sqrt(sum((c - ctr) ** 2 for c, ctr in zip(coords, center)))
        mask = dist < 18
        vol[mask] += 0.7
        label[mask] = 1
        vols.append(vol)
        labels.append(label)
    return (
        torch.from_numpy(np.stack(vols)[:, None]),  # (N, 1, D, H, W)
        torch.from_numpy(np.stack(labels)),  # (N, D, H, W)
    )


x_train, y_train = make_batch(N_SAMPLES, SPATIAL)
print(f"Training data: x={x_train.shape}, y={y_train.shape}")

# %% [markdown]
# ## Build and train the model

# %%
model = unet(n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses = []
model.train()
for epoch in range(50):
    epoch_loss = 0.0
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        xb = x_train[i : i + BATCH_SIZE].to(device)
        yb = y_train[i : i + BATCH_SIZE].to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (N_SAMPLES / BATCH_SIZE)
    losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:3d}: loss={avg_loss:.4f}")

# %% [markdown]
# ## Evaluate with predict()

# %%
model.eval()
test_vol = x_train[0, 0].numpy()  # single volume
test_label = y_train[0].numpy()

result = predict(
    inputs=test_vol,
    model=model,
    block_shape=(32, 32, 32),
    batch_size=4,
    device=str(device),
    return_labels=True,
)

pred_arr = np.asarray(result.dataobj)
pred_bin = (pred_arr == 1).astype(np.float32)
label_bin = (test_label == 1).astype(np.float32)

intersection = (pred_bin * label_bin).sum()
dice = 2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)  # noqa: E226
print(f"Dice score: {dice:.4f}")

# %% [markdown]
# ## Save the trained model

# %%
torch.save(model.state_dict(), "unet_trained.pth")
print("Model saved to unet_trained.pth")

# %% [markdown]
# ## Using real NIfTI data
#
# Replace the synthetic data with real files using the MONAI-backed
# data pipeline:
#
# ```python
# from nobrainer.dataset import get_dataset
#
# data_files = [
#     {"image": "sub-001_T1w.nii.gz", "label": "sub-001_label.nii.gz"},
#     {"image": "sub-002_T1w.nii.gz", "label": "sub-002_label.nii.gz"},
#     ...
# ]
# loader = get_dataset(data=data_files, batch_size=2, augment=True, cache=True)
#
# for batch in loader:
#     images = batch["image"].to(device)
#     labels = batch["label"].to(device)
#     ...
# ```
