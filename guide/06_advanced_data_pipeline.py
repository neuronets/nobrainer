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
# # Advanced Data Pipeline
#
# This tutorial covers advanced data-handling features:
#
# 1. Custom MONAI transforms with the Dataset builder
# 2. Converting NIfTI to Zarr v3 with `nobrainer.io.nifti_to_zarr`
# 3. Exporting Croissant-ML dataset metadata
# 4. Multi-GPU training via `nobrainer.training.fit()`
#
# Each section is self-contained so you can run them independently.

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
            "nobrainer[zarr,dev]",
            "monai",
            "nilearn",
            "matplotlib",
        ]
        + (["--pre"] if PRE_RELEASE else []),
        check=True,
    )
except ImportError:
    pass

# %%
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from nobrainer.io import read_csv  # noqa: E402
from nobrainer.processing import Dataset  # noqa: E402
from nobrainer.utils import get_data  # noqa: E402

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subject pairs")

# %% [markdown]
# ## 1. Custom MONAI transforms
#
# The `Dataset` builder integrates with the MONAI data pipeline.
# You can also compose your own transforms and feed a standard
# PyTorch DataLoader to the estimator's `.fit()`.

# %%
try:
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        RandAffined,
        RandFlipd,
        ScaleIntensityd,
        ToTensord,
    )

    keys = ["image", "label"]
    train_transforms = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityd(keys=["image"]),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandAffined(
                keys=keys,
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                mode=("bilinear", "nearest"),
            ),
            ToTensord(keys=keys),
        ]
    )
    print("MONAI transforms created:")
    for t in train_transforms.transforms:
        print(f"  - {type(t).__name__}")

except ImportError:
    print("MONAI not installed — skipping custom transform demo")
    train_transforms = None

# %% [markdown]
# With a custom transform pipeline you would pass a standard DataLoader
# directly to the estimator:
#
# ```python
# from monai.data import CacheDataset, DataLoader
#
# data_dicts = [{"image": p[0], "label": p[1]} for p in filepaths]
# monai_ds = CacheDataset(data_dicts, transform=train_transforms)
# loader = DataLoader(monai_ds, batch_size=2, shuffle=True)
#
# seg = Segmentation("unet", model_args={...})
# seg.fit(loader, epochs=10)  # accepts any DataLoader
# ```

# %% [markdown]
# ## 2. NIfTI to Zarr conversion
#
# `nifti_to_zarr` converts a NIfTI file to chunked Zarr v3 format,
# which enables efficient random-access reads for large volumes.

# %%
from nobrainer.io import nifti_to_zarr  # noqa: E402

zarr_dir = Path("/tmp/nobrainer_zarr_demo")
zarr_dir.mkdir(parents=True, exist_ok=True)

img_path = filepaths[0][0]
zarr_path = zarr_dir / "subject_00.zarr"

nifti_to_zarr(img_path, zarr_path, chunk_shape=(64, 64, 64))
print(f"Converted: {img_path}")
print(f"Zarr output: {zarr_path}")

# Inspect the Zarr store
import zarr  # noqa: E402

z = zarr.open(str(zarr_path), mode="r")
print(f"Zarr shape: {z.shape}, dtype: {z.dtype}, chunks: {z.chunks}")

# %% [markdown]
# ## 3. Export Croissant-ML dataset metadata
#
# `Dataset.to_croissant()` generates a JSON-LD file following the
# [Croissant-ML](https://mlcommons.org/croissant/) standard.

# %%
ds = Dataset.from_files(filepaths, block_shape=(32, 32, 32), n_classes=2)

croissant_path = zarr_dir / "dataset_croissant.json"
ds.to_croissant(croissant_path)

meta = json.loads(croissant_path.read_text())
print(json.dumps(meta, indent=2)[:500] + "\n...")

# %% [markdown]
# ## 4. Multi-GPU training with `nobrainer.training.fit()`
#
# The low-level `training.fit()` function supports Distributed Data
# Parallel (DDP) when multiple GPUs are available.  The estimator
# API uses this internally when `multi_gpu=True` (the default).
#
# Below is the explicit multi-GPU pattern.  On a single-GPU or CPU
# machine it falls back gracefully.

# %%
from nobrainer.models.segmentation import unet  # noqa: E402
from nobrainer.training import fit as training_fit  # noqa: E402

model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Build a DataLoader from the Dataset builder
loader = ds.batch(2).dataloader

gpus = torch.cuda.device_count()
print(f"Available GPUs: {gpus}")

result = training_fit(
    model=model,
    loader=loader,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=2,
    gpus=max(gpus, 1),
)

print(f"Training complete: {result}")

# %% [markdown]
# ## Summary
#
# | Feature | Module |
# |---------|--------|
# | Custom transforms | `monai.transforms` + any DataLoader |
# | Zarr conversion | `nobrainer.io.nifti_to_zarr()` |
# | Dataset metadata | `Dataset.to_croissant()` |
# | Multi-GPU training | `nobrainer.training.fit(gpus=N)` |
# | Estimator shortcut | `Segmentation(..., multi_gpu=True).fit()` |
