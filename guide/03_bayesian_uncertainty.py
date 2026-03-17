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
# # Bayesian Inference with Uncertainty Maps (Estimator API)
#
# This tutorial uses a Bayesian VNet via the estimator API to obtain
# segmentation **with uncertainty quantification**.
#
# 1. Download a real T1-weighted brain volume
# 2. Train a small Bayesian VNet with `Segmentation.fit()`
# 3. Run Monte-Carlo inference with `n_samples=5`
# 4. Inspect label, variance, and entropy outputs

# %%
# Colab install (uncomment if needed)
PRE_RELEASE = False
try:
    import subprocess

    import google.colab  # noqa: F401

    subprocess.run(
        [
            "pip",
            "install",
            "-q",
            "nobrainer[bayesian,dev]",
            "monai",
            "pyro-ppl",
            "nilearn",
            "matplotlib",
        ]
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
# ## Load real brain MRI data

# %%
csv_path = get_data()
filepaths = read_csv(csv_path)
vol_path, label_path = filepaths[0]
print(f"Volume: {vol_path}")

# %% [markdown]
# ## Build dataset and train
#
# We use only 1 subject with a small block shape for a quick demo.
# The `bayesian_vnet` uses Pyro-based stochastic weight layers so each
# forward pass samples different weights.

# %%
ds = (
    Dataset.from_files(filepaths[:1], block_shape=(32, 32, 32), n_classes=2)
    .batch(2)
    .binarize()
)

seg = Segmentation(
    "bayesian_vnet",
    model_args={
        "in_channels": 1,
        "n_classes": 2,
        "base_filters": 8,
        "levels": 2,
    },
).fit(ds, epochs=10)

# %% [markdown]
# ## Monte-Carlo prediction with uncertainty
#
# Passing `n_samples > 0` to `.predict()` triggers stochastic forward
# passes.  The method returns a tuple of NIfTI images:
# - **label** — argmax of mean softmax probabilities
# - **variance** — mean predictive variance across classes
# - **entropy** — predictive entropy of the mean distribution

# %%
label_img, var_img, entropy_img = seg.predict(
    vol_path, block_shape=(32, 32, 32), n_samples=5
)

print(f"Label shape:    {label_img.shape}")
print(f"Variance shape: {var_img.shape}")
print(f"Entropy shape:  {entropy_img.shape}")

# %%
var_arr = np.asarray(var_img.dataobj)
entropy_arr = np.asarray(entropy_img.dataobj)

# Dice against binarized ground truth
gt = (np.asarray(nib.load(label_path).dataobj) > 0).astype(np.float32)
pred = (np.asarray(label_img.dataobj) == 1).astype(np.float32)
dice = 2 * (pred * gt).sum() / (pred.sum() + gt.sum() + 1e-8)

print(f"Dice score:     {dice:.4f}")
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
