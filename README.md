# Nobrainer

![Build status](https://github.com/neuronets/nobrainer/actions/workflows/ci.yml/badge.svg)

_Nobrainer_ is a deep learning framework for 3D brain image processing built on
**PyTorch** and **MONAI**. It provides segmentation models (deterministic and
Bayesian), generative models, a MONAI-native data pipeline, block-based
prediction with uncertainty quantification, and CLI tools for inference and
automated hyperparameter search.

Pre-trained models for brain extraction, segmentation, and generation are
available in the [trained-models](https://github.com/neuronets/trained-models)
repository.

The _Nobrainer_ project is supported by NIH RF1MH121885 and is distributed
under the Apache 2.0 license.

## Models

### Segmentation

| Model | Backend | Application |
|-------|---------|-------------|
| [UNet](nobrainer/models/segmentation.py) | MONAI | segmentation |
| [VNet](nobrainer/models/segmentation.py) | MONAI | segmentation |
| [Attention U-Net](nobrainer/models/segmentation.py) | MONAI | segmentation |
| [UNETR](nobrainer/models/segmentation.py) | MONAI | segmentation |
| [MeshNet](nobrainer/models/meshnet.py) | PyTorch | segmentation |
| [HighResNet](nobrainer/models/highresnet.py) | PyTorch | segmentation |

### Bayesian (uncertainty quantification)

| Model | Backend | Application |
|-------|---------|-------------|
| [Bayesian VNet](nobrainer/models/bayesian/bayesian_vnet.py) | Pyro | segmentation + uncertainty |
| [Bayesian MeshNet](nobrainer/models/bayesian/bayesian_meshnet.py) | Pyro | segmentation + uncertainty |

### Generative

| Model | Backend | Application |
|-------|---------|-------------|
| [Progressive GAN](nobrainer/models/generative/progressivegan.py) | PyTorch Lightning | brain generation |
| [DCGAN](nobrainer/models/generative/dcgan.py) | PyTorch Lightning | brain generation |

### Other

| Model | Application |
|-------|-------------|
| [Autoencoder](nobrainer/models/autoencoder.py) | representation learning |
| [SimSiam](nobrainer/models/simsiam.py) | self-supervised learning |

### Custom layers

- `BernoulliDropout`, `ConcreteDropout`, `GaussianDropout` — stochastic regularization
- `BayesianConv3d`, `BayesianLinear` — Pyro-based weight uncertainty layers
- `MaxPool4D` — 4D max pooling via reshape

### Losses and metrics

**Losses**: Dice, Generalized Dice, Jaccard, Tversky, ELBO (Bayesian), Wasserstein, Gradient Penalty

**Metrics**: Dice, Jaccard, Hausdorff distance (all via MONAI)

## Installation

### pip / uv

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install nobrainer
```

For Bayesian and generative model support:

```bash
uv pip install "nobrainer[bayesian,generative]" monai pyro-ppl
```

### Docker

GPU image (requires NVIDIA driver on host):

```bash
docker pull neuronets/nobrainer:latest-gpu-pt
docker run --gpus all --rm neuronets/nobrainer:latest-gpu-pt predict --help
```

CPU-only image:

```bash
docker pull neuronets/nobrainer:latest-cpu-pt
docker run --rm neuronets/nobrainer:latest-cpu-pt predict --help
```

## Quick start

### Tutorials

See the [Nobrainer Book](https://neuronets.dev/nobrainer-book/) for 11
progressive tutorials — from installation to contributing.

### sr-tests (somewhat realistic tests)

`nobrainer/sr-tests/` contains pytest integration tests that exercise the
real API with real brain data. They run in CI on every push:

```bash
pytest nobrainer/sr-tests/ -v -m "not gpu" --tb=short
```

### Simple API (3 lines)

```python
from nobrainer.processing import Segmentation, Dataset

ds = Dataset.from_files(filepaths, block_shape=(128, 128, 128), n_classes=2).batch(2)
result = Segmentation("unet").fit(ds, epochs=5).predict("brain.nii.gz")
```

Models are saved with [Croissant-ML](https://mlcommons.org/croissant/) metadata
for reproducibility:

```python
seg.save("my_model")  # Creates model.pth + croissant.json
seg = Segmentation.load("my_model")
```

### Brain segmentation (CLI)

```bash
nobrainer predict \
  --model unet_brainmask.pth \
  --model-type unet \
  --n-classes 2 \
  input_T1w.nii.gz output_mask.nii.gz
```

### Brain segmentation (Python)

```python
import torch
import nobrainer
from nobrainer.prediction import predict

model = nobrainer.models.unet(n_classes=2)
model.load_state_dict(torch.load("unet_brainmask.pth"))
model.eval()

result = predict(
    inputs="input_T1w.nii.gz",
    model=model,
    block_shape=(128, 128, 128),
    device="cuda",
)
result.to_filename("output_mask.nii.gz")
```

### Bayesian inference with uncertainty maps

```python
from nobrainer.prediction import predict_with_uncertainty

model = nobrainer.models.bayesian_vnet(n_classes=2)
model.load_state_dict(torch.load("bayesian_vnet.pth"))

label, variance, entropy = predict_with_uncertainty(
    inputs="input_T1w.nii.gz",
    model=model,
    n_samples=10,
    block_shape=(128, 128, 128),
    device="cuda",
)
label.to_filename("label.nii.gz")
variance.to_filename("variance.nii.gz")
entropy.to_filename("entropy.nii.gz")
```

### Brain generation

```bash
nobrainer generate \
  --model progressivegan.ckpt \
  --model-type progressivegan \
  output_synthetic.nii.gz
```

### Zarr v3 data pipeline

```python
from nobrainer.io import nifti_to_zarr, zarr_to_nifti

# Convert NIfTI to sharded Zarr v3 with multi-resolution pyramid
nifti_to_zarr("brain_T1w.nii.gz", "brain.zarr", chunk_shape=(64, 64, 64), levels=3)

# Load Zarr stores directly in the training pipeline
from nobrainer.dataset import get_dataset

loader = get_dataset(
    data=[{"image": "brain.zarr", "label": "label.zarr"}],
    batch_size=2,
)

# Round-trip back to NIfTI
zarr_to_nifti("brain.zarr", "brain_roundtrip.nii.gz")
```

### Training a model

```python
import torch
from nobrainer.dataset import get_dataset
from nobrainer.losses import dice

data_files = [
    {"image": f"sub-{i:03d}_T1w.nii.gz", "label": f"sub-{i:03d}_label.nii.gz"}
    for i in range(1, 101)
]
loader = get_dataset(data=data_files, batch_size=2, augment=True, cache=True)

model = nobrainer.models.unet(n_classes=2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = dice()

for epoch in range(50):
    model.train()
    for batch in loader:
        images, labels = batch["image"].cuda(), batch["label"].cuda()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "unet_trained.pth")
```

## Automated research (autoresearch)

Nobrainer includes an automated hyperparameter search loop that uses an LLM
to propose training modifications overnight:

```bash
nobrainer research run \
  --working-dir ./research/bayesian_vnet \
  --model-family bayesian_vnet \
  --max-experiments 15 \
  --budget-hours 8
```

Improved models are versioned via DataLad:

```bash
nobrainer research commit \
  --run-dir ./research/bayesian_vnet \
  --trained-models-path ~/trained-models \
  --model-family bayesian_vnet
```

## GPU test dispatch (nobrainer-runner)

[nobrainer-runner](https://github.com/neuronets/nobrainer-runner) submits GPU
test suites to Slurm clusters or cloud instances (AWS Batch, GCP Batch):

```bash
nobrainer-runner submit --profile mycluster --gpus 1 "pytest tests/ -m gpu"
nobrainer-runner status $JOB_ID
nobrainer-runner results --format json $JOB_ID
```

## Package layout

- `nobrainer.models` — segmentation, Bayesian, and generative `torch.nn.Module` models
- `nobrainer.losses` — Dice, Jaccard, Tversky, ELBO, Wasserstein (MONAI-backed)
- `nobrainer.metrics` — Dice, Jaccard, Hausdorff (MONAI-backed)
- `nobrainer.dataset` — MONAI `CacheDataset` + `DataLoader` pipeline
- `nobrainer.prediction` — block-based `predict()` and `predict_with_uncertainty()`
- `nobrainer.io` — `convert_tfrecords()`, `convert_weights()` (TF → PyTorch migration)
- `nobrainer.layers` — dropout layers, Bayesian layers, MaxPool4D
- `nobrainer.research` — autoresearch loop and DataLad model versioning
- `nobrainer.cli` — Click CLI (`predict`, `generate`, `research`, `commit`, `info`)

## Development and releases

Nobrainer uses a two-branch release workflow:

| Branch | Purpose | PyPI version |
|--------|---------|--------------|
| `master` | Stable releases | `uv pip install nobrainer` |
| `alpha` | Pre-releases for testing | `uv pip install --pre nobrainer` |

**Alpha workflow**: Feature branches merge to `alpha`. Each merge triggers
book tutorial validation (using a matching branch on
[nobrainer-book](https://github.com/neuronets/nobrainer-book) if available,
otherwise the book's `alpha` branch) followed by an automatic pre-release
tag (e.g., `0.5.0-alpha.0`).

**Stable workflow**: When `alpha` is merged to `master` with the `release`
label, a stable version is tagged and published to PyPI.

**GPU CI**: PRs to `master` can request GPU testing on EC2 by adding the
`gpu-test-approved` label. Instance type and spot pricing are configurable
via `gpu-instance:<type>` and `gpu-spot:true` labels.

## Citation

If you use this package, please [cite](https://github.com/neuronets/nobrainer/blob/master/CITATION) it.

## Questions or issues

Please [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose).
