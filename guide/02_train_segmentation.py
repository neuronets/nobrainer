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
# # Train a Segmentation Model (Estimator API)
#
# This tutorial trains a UNet for binary brain segmentation using the
# scikit-learn-style **estimator API**.
#
# 1. Download sample brain data (10 T1w/label pairs)
# 2. Prepare training data with `extract_patches()`
# 3. Build a `Dataset` with the fluent builder
# 4. Train with `Segmentation.fit()` — one line
# 5. Predict and compute Dice score
# 6. Save with Croissant-ML metadata

# %%
# Colab install
PRE_RELEASE = False
try:
    import subprocess

    import google.colab  # noqa: F401

    subprocess.run(
        ["pip", "install", "-q", "nobrainer[dev]", "monai", "nilearn", "matplotlib"]
        + (["--pre"] if PRE_RELEASE else []),
        check=True,
    )
except ImportError:
    pass

# %%
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from nobrainer.io import read_csv  # noqa: E402
from nobrainer.processing import Segmentation, extract_patches  # noqa: E402
from nobrainer.utils import get_data  # noqa: E402

# %% [markdown]
# ## 1. Download sample brain MRI data
#
# `get_data()` downloads 10 T1w / FreeSurfer aparc+aseg pairs (~46 MB).
# We hold out 1 subject for evaluation.

# %%
csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subject pairs")

train_pairs = filepaths[:9]
eval_pair = filepaths[9]

# %% [markdown]
# ## 2. Prepare training data
#
# The raw volumes are ~256^3 — too large for CPU training. We extract
# small random 32^3 patches using `extract_patches()`.  The `binarize=True`
# option converts FreeSurfer parcellation labels (like region 2007) into a
# binary brain mask (0 = background, 1 = brain).
#
# For custom label selection, you can pass a set of label IDs or a function:
# ```python
# # Select only hippocampus (FreeSurfer labels 17, 53)
# extract_patches(vol, lbl, binarize={17, 53})
#
# # Custom: cortical regions only
# extract_patches(vol, lbl, binarize=lambda x: (x >= 1000).astype(float))
# ```

# %%
BLOCK_SHAPE = (32, 32, 32)
N_PATCHES = 2  # per volume (use more for real training)

all_patches = []
for img_path, label_path in train_pairs:
    patches = extract_patches(
        img_path,
        label_path,
        block_shape=BLOCK_SHAPE,
        n_patches=N_PATCHES,
        binarize=True,
    )
    all_patches.extend(patches)

print(f"Extracted {len(all_patches)} patches from {len(train_pairs)} subjects")
print(f"Each patch: image {all_patches[0][0].shape}, label {all_patches[0][1].shape}")

# Check label distribution in a sample patch
sample_lbl = all_patches[0][1]
print(f"Label values: {np.unique(sample_lbl)} (0=background, 1=brain)")

# %% [markdown]
# ## 3. Build the Dataset
#
# Two approaches to create a training dataset:
#
# **Option A — From files (MONAI pipeline handles patching):**
# ```python
# ds = Dataset.from_files(train_pairs, block_shape=(32,32,32), n_classes=2)
#     .batch(2).augment().binarize()
# ```
#
# **Option B — From pre-extracted patches (more control):**
# Build a `torch.utils.data.DataLoader` from the patches directly.

# %%
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

x = torch.from_numpy(np.stack([p[0] for p in all_patches])[:, None])  # (N,1,D,H,W)
y = torch.from_numpy(np.stack([p[1] for p in all_patches])).long()  # (N,D,H,W)
loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=True)

print(f"Training: {x.shape[0]} patches, batch_size=2")

# %% [markdown]
# ## 4. Train with the Estimator API
#
# `Segmentation.fit()` accepts either a `Dataset` object or a raw
# `DataLoader`.  Here we pass the pre-built loader.

# %%
seg = Segmentation("unet", model_args={"channels": (8, 16, 32), "strides": (2, 2)}).fit(
    loader, epochs=5
)

# %% [markdown]
# ## 5. Predict and evaluate
#
# `.predict()` runs block-based inference on the held-out subject.

# %%
eval_img_path, eval_label_path = eval_pair
result = seg.predict(eval_img_path, block_shape=BLOCK_SHAPE)

pred_arr = np.asarray(result.dataobj)
label_arr = np.asarray(nib.load(eval_label_path).dataobj, dtype=np.float32)
label_bin = (label_arr > 0).astype(np.float32)
pred_bin = (pred_arr == 1).astype(np.float32)

intersection = (pred_bin * label_bin).sum()
dice = 2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)  # noqa: E226
print(f"Dice score on held-out subject: {dice:.4f}")
print("(Low because we trained briefly on small patches)")

# %% [markdown]
# ## 6. Save with Croissant-ML metadata
#
# `.save()` writes `model.pth` and a `croissant.json` that records the
# model architecture, optimizer, loss, training data checksums, and
# software versions — everything needed for reproducibility.

# %%
import json  # noqa: E402
from pathlib import Path  # noqa: E402

save_dir = Path("/tmp/nobrainer_seg_demo")
seg.save(save_dir)

croissant = json.loads((save_dir / "croissant.json").read_text())
print(json.dumps(croissant, indent=2)[:600] + "\n...")
