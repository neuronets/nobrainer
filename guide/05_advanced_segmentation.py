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
# # Advanced Segmentation — Raw PyTorch API
#
# This tutorial achieves the **same result** as `02_train_segmentation.py`
# but uses the lower-level PyTorch API directly.  Each section includes
# a comment showing the equivalent estimator-API call.
#
# Use this approach when you need full control over the training loop,
# custom losses, learning-rate schedules, or mixed-precision training.

# %%
# Colab install (uncomment if needed)
PRE_RELEASE = False
try:
    import subprocess

    import google.colab  # noqa: F401

    subprocess.run(
        ["pip", "install", "nobrainer" + ("[dev]" if PRE_RELEASE else "")],
        check=True,
    )
except ImportError:
    pass

# %%
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from nobrainer.io import read_csv  # noqa: E402
from nobrainer.utils import get_data  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Download and split data
#
# **Estimator equivalent:**
# ```python
# ds = Dataset.from_files(filepaths, block_shape=(32,32,32), n_classes=2)
# ```

# %%
torch.manual_seed(42)
np.random.seed(42)

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subject pairs")

train_pairs = filepaths[:9]
eval_pair = filepaths[9]

# %% [markdown]
# ## 2. Extract random patches
#
# The full volumes are ~256^3 — too large for CPU training.  We pull
# small random 32^3 blocks from each.  The estimator `Dataset` does
# this automatically via the MONAI data pipeline.

# %%
BLOCK = 32
N_PATCHES_PER_VOL = 2


def extract_random_patches(img_path, label_path, block=BLOCK, n=2):
    """Load a volume pair and extract random cubic patches."""
    vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(label_path).dataobj, dtype=np.int64)
    lab = (lab > 0).astype(np.int64)  # binarize

    xs, ys = [], []
    for _ in range(n):
        starts = [np.random.randint(0, max(s - block, 1)) for s in vol.shape[:3]]
        sl = tuple(slice(s, s + block) for s in starts)
        xs.append(vol[sl])
        ys.append(lab[sl])
    return xs, ys


all_x, all_y = [], []
for img_path, label_path in train_pairs:
    px, py = extract_random_patches(img_path, label_path)
    all_x.extend(px)
    all_y.extend(py)

x_train = torch.from_numpy(np.stack(all_x)[:, None])  # (N,1,D,H,W)
y_train = torch.from_numpy(np.stack(all_y))  # (N,D,H,W)
print(f"Training patches: x={x_train.shape}, y={y_train.shape}")

# %% [markdown]
# ## 3. Build the model
#
# **Estimator equivalent:**
# ```python
# seg = Segmentation("unet", model_args={
#     "channels": (8, 16, 32), "strides": (2, 2)
# })
# ```

# %%
from nobrainer.models.segmentation import unet  # noqa: E402

model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2))
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model.__class__.__name__}  ({n_params:,} params)")

# %% [markdown]
# ## 4. Training loop
#
# **Estimator equivalent:**
# ```python
# seg.fit(ds_train, epochs=5)
# ```

# %%
BATCH_SIZE = 4
N_SAMPLES = x_train.shape[0]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    epoch_loss = 0.0
    n_batches = 0
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
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}: loss={avg_loss:.4f}")

# %% [markdown]
# ## 5. Predict and evaluate
#
# **Estimator equivalent:**
# ```python
# result = seg.predict(eval_vol)
# ```

# %%
from nobrainer.prediction import predict  # noqa: E402

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
dice = 2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)
print(f"Dice score on held-out subject: {dice:.4f}")
print("(Same result as 02_train_segmentation.py)")

# %% [markdown]
# ## Summary: Estimator vs Raw API
#
# | Step | Estimator API | Raw PyTorch |
# |------|--------------|-------------|
# | Data | `Dataset.from_files(...).batch(2).augment()` | Manual patch extraction + DataLoader |
# | Model | `Segmentation("unet", ...)` | `from nobrainer.models.segmentation import unet` |
# | Train | `.fit(ds, epochs=5)` | Explicit optimizer + loss loop |
# | Predict | `.predict(path)` | `from nobrainer.prediction import predict` |
# | Save | `.save("dir")` | `torch.save(model.state_dict(), ...)` |
