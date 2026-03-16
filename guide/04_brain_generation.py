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
# # Train a Progressive GAN for Brain Generation (Estimator API)
#
# This tutorial trains a Progressive GAN on real brain MRI data using
# the scikit-learn-style **Generation** estimator.  Volumes are
# downsampled to 4^3 resolution for fast CPU training.
#
# 1. Download and prepare training data
# 2. One-line GAN training with `Generation.fit()`
# 3. Generate synthetic brain volumes with `.generate()`

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
from scipy.ndimage import zoom  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from nobrainer.io import read_csv  # noqa: E402
from nobrainer.processing import Generation  # noqa: E402
from nobrainer.utils import get_data  # noqa: E402

# %% [markdown]
# ## Download and downsample real brain MRI data
#
# We load all 10 T1-weighted volumes, normalize each to [0, 1], and
# downsample to 4^3 for fast GAN training on CPU.

# %%
TARGET_SIZE = 4

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subjects")

volumes = []
for img_path, _ in filepaths:
    vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    factors = [TARGET_SIZE / s for s in vol.shape[:3]]
    volumes.append(zoom(vol, factors, order=1))

imgs = torch.from_numpy(np.stack(volumes)[:, None])  # (N, 1, 4, 4, 4)
print(f"Training set: {imgs.shape[0]} volumes of shape {TARGET_SIZE}^3")

# %%
torch.manual_seed(42)
loader = DataLoader(TensorDataset(imgs), batch_size=4, shuffle=True)

# %% [markdown]
# ## Train the ProgressiveGAN
#
# We use a single resolution level (4) and minimal feature maps to
# keep this demo fast on CPU.  The `Generation` estimator wraps
# Lightning training internally.

# %%
gen = Generation(
    "progressivegan",
    model_args={
        "latent_size": 64,
        "fmap_base": 64,
        "fmap_max": 64,
        "resolution_schedule": [4],
        "steps_per_phase": 500,
    },
).fit(loader, epochs=100)

# %% [markdown]
# ## Generate synthetic volumes
#
# `.generate()` returns a list of NIfTI images.

# %%
synth = gen.generate(n_images=4)

for i, img in enumerate(synth):
    arr = np.asarray(img.dataobj)
    print(
        f"  Volume {i}: shape={arr.shape}  " f"range=[{arr.min():.3f}, {arr.max():.3f}]"
    )

# %% [markdown]
# ## Save and reload
#
# ```python
# gen.save("my_gan")
# gen2 = Generation.load("my_gan")
# new_images = gen2.generate(n_images=2)
# ```
