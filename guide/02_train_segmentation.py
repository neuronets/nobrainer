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
# 2. Build a `Dataset` with the fluent builder
# 3. Train with `Segmentation.fit()` — one line
# 4. Predict and compute Dice score
# 5. Save with Croissant-ML metadata

# %%
# Colab install (uncomment if needed)
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
from nobrainer.processing import Dataset, Segmentation  # noqa: E402
from nobrainer.utils import get_data  # noqa: E402

# %% [markdown]
# ## Download and prepare data
#
# `get_data()` fetches 10 T1w / aparc+aseg pairs.  The labels are
# automatically binarized (0 = background, 1 = brain) by the dataset
# pipeline.  We hold out 1 subject for evaluation.

# %%
csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subject pairs")

train_pairs = filepaths[:9]
eval_pair = filepaths[9]

# %% [markdown]
# ## Build the Dataset
#
# `Dataset.from_files()` accepts the list of (image, label) tuples.
# Random 32^3 patches are extracted from the ~256^3 volumes at train
# time.  Chaining `.batch()`, `.augment()`, and `.binarize()` configures
# the pipeline.  `.binarize()` converts multi-label parcellations to
# binary brain masks (any non-zero label → 1).

# %%
ds_train = (
    Dataset.from_files(train_pairs, block_shape=(32, 32, 32), n_classes=2)
    .batch(2)
    .augment()
    .binarize()
)

# %% [markdown]
# ## Train
#
# One line: create an estimator and call `.fit()`.

# %%
seg = Segmentation("unet", model_args={"channels": (8, 16, 32), "strides": (2, 2)}).fit(
    ds_train, epochs=5
)

# %% [markdown]
# ## Predict and evaluate
#
# `.predict()` runs block-based inference and returns a NIfTI image.

# %%
eval_img_path, eval_label_path = eval_pair
result = seg.predict(eval_img_path, block_shape=(32, 32, 32))

pred_arr = np.asarray(result.dataobj)
label_arr = np.asarray(nib.load(eval_label_path).dataobj, dtype=np.float32)
label_bin = (label_arr > 0).astype(np.float32)
pred_bin = (pred_arr == 1).astype(np.float32)

intersection = (pred_bin * label_bin).sum()
dice = 2 * intersection / (pred_bin.sum() + label_bin.sum() + 1e-8)
print(f"Dice score on held-out subject: {dice:.4f}")
print("(Low because we trained briefly on small patches)")

# %% [markdown]
# ## Save with Croissant-ML metadata
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
