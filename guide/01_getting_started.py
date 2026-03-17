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
# # Getting Started with Nobrainer
#
# Nobrainer is a deep learning framework for 3D brain image processing.
# This tutorial shows the **estimator API** — a scikit-learn-style
# interface that lets you train and predict in just a few lines.
#
# 1. Download sample brain MRI data
# 2. Run 3-line segmentation on a real brain volume
# 3. List available model families

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
import torch  # noqa: E402

from nobrainer.processing import Dataset, Segmentation  # noqa: E402
from nobrainer.utils import get_data  # noqa: E402

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

# %% [markdown]
# ## Download sample brain MRI data
#
# `get_data()` downloads 10 T1-weighted / FreeSurfer label pairs (~46 MB)
# and returns a CSV path.  `read_csv()` parses it into path tuples.

# %%
from nobrainer.io import read_csv  # noqa: E402

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Number of subjects: {len(filepaths)}")
print(f"First pair:\n  image: {filepaths[0][0]}\n  label: {filepaths[0][1]}")

# %% [markdown]
# ## Three-line segmentation
#
# 1. Build a `Dataset` from the downloaded files
# 2. Create a `Segmentation` estimator and `.fit()` it
# 3. `.predict()` on an evaluation volume
#
# The model is tiny (few channels) so it runs in seconds on CPU.

# %%
ds = (
    Dataset.from_files(filepaths, block_shape=(32, 32, 32), n_classes=2)
    .batch(2)
    .augment()
    .binarize()
)

seg = Segmentation("unet", model_args={"channels": (8, 16, 32), "strides": (2, 2)})
seg.fit(ds, epochs=2)

result = seg.predict(filepaths[0][0], block_shape=(32, 32, 32))
print(f"Prediction shape: {result.shape}")

# %% [markdown]
# The model was only trained for 2 epochs on small patches, so
# predictions will not be meaningful.  See `02_train_segmentation.py`
# for a complete training loop with evaluation.

# %% [markdown]
# ## Available models
#
# Nobrainer includes segmentation, Bayesian, and generative families:

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
