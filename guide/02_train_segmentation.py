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
# using real T1-weighted MRI data from `nobrainer.utils.get_data()`.
#
# We will:
# 1. Download sample brain data (10 T1w/label pairs)
# 2. Build a small UNet model
# 3. Train on random 3-D patches extracted from the volumes
# 4. Evaluate with the predict() API and compute Dice score

# %%
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from nobrainer.io import read_csv
from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict
from nobrainer.utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Download and load real brain MRI data
#
# `get_data()` downloads 10 T1w / aparc+aseg pairs. We binarize the
# labels (0 = background, 1 = brain) for a simple brain-mask task.

# %%
torch.manual_seed(42)
np.random.seed(42)

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subject pairs")

# Split: 9 for training, 1 for evaluation
train_pairs = filepaths[:9]
eval_pair = filepaths[9]

# %% [markdown]
# ## Extract random patches for training
#
# The full volumes are ~256^3 which is too large for CPU training.
# We extract small random patches from each volume.

# %%
BLOCK = 32
N_PATCHES_PER_VOL = 2


def extract_random_patches(
    img_path, label_path, block=BLOCK, n_patches=N_PATCHES_PER_VOL
):
    """Load a volume pair and extract random cubic patches."""
    img = nib.load(img_path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(label_path).dataobj, dtype=np.int64)
    # Binarize: any label > 0 is brain
    lab = (lab > 0).astype(np.int64)

    patches_x, patches_y = [], []
    for _ in range(n_patches):
        # Random start within valid range
        starts = [np.random.randint(0, max(s - block, 1)) for s in vol.shape[:3]]
        sl = tuple(slice(s, s + block) for s in starts)
        patches_x.append(vol[sl])
        patches_y.append(lab[sl])
    return patches_x, patches_y


all_x, all_y = [], []
for img_path, label_path in train_pairs:
    px, py = extract_random_patches(img_path, label_path)
    all_x.extend(px)
    all_y.extend(py)

x_train = torch.from_numpy(np.stack(all_x)[:, None])  # (N, 1, D, H, W)
y_train = torch.from_numpy(np.stack(all_y))  # (N, D, H, W)
print(f"Training patches: x={x_train.shape}, y={y_train.shape}")

# %% [markdown]
# ## Build and train the model
#
# We use a small UNet (3 levels, narrow channels) to keep this fast.

# %%
BATCH_SIZE = 4
N_SAMPLES = x_train.shape[0]

model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses = []
model.train()
for epoch in range(20):
    epoch_loss = 0.0
    n_batches = 0
    # Shuffle each epoch
    perm = torch.randperm(N_SAMPLES)
    for i in range(0, N_SAMPLES, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        xb = x_train[idx].to(device)
        yb = y_train[idx].to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1:3d}: loss={avg_loss:.4f}")

# %% [markdown]
# ## Evaluate with predict()
#
# We run block-based prediction on the held-out subject and compute the
# Dice coefficient for the binary brain mask.

# %%
model.eval()
eval_img_path, eval_label_path = eval_pair

result = predict(
    inputs=eval_img_path,
    model=model,
    block_shape=(32, 32, 32),
    batch_size=4,
    device=str(device),
    return_labels=True,
)

pred_arr = np.asarray(result.dataobj)
label_arr = np.asarray(nib.load(eval_label_path).dataobj, dtype=np.float32)
label_bin = (label_arr > 0).astype(np.float32)
pred_bin = (pred_arr == 1).astype(np.float32)

intersection = (pred_bin * label_bin).sum()
dice = 2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)  # noqa: E226
print(f"Dice score on held-out subject: {dice:.4f}")
print("(Note: score is low because we trained briefly on small patches)")

# %% [markdown]
# ## Next steps
#
# For better results:
# - Train longer with more patches per volume
# - Use the MONAI-backed `nobrainer.dataset.get_dataset()` for proper
#   data loading with augmentation and caching
# - Use larger block shapes and more model capacity
#
# ```python
# from nobrainer.dataset import get_dataset
#
# images = [p[0] for p in filepaths]
# labels = [p[1] for p in filepaths]
# loader = get_dataset(
#     image_paths=images,
#     label_paths=labels,
#     block_shape=(64, 64, 64),
#     batch_size=2,
#     augment=True,
# )
# ```
