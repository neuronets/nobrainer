# KWYK Brain Extraction Reproduction

Reproduce the kwyk brain extraction study (McClure et al., Frontiers in
Neuroinformatics 2019) using the refactored PyTorch nobrainer.

## Quick Start

```bash
# 1. Create venv and install
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e "../../[bayesian,versioning,dev]" monai pyro-ppl datalad matplotlib pyyaml

# 2. Assemble dataset
python 01_assemble_dataset.py --datasets ds000114 --output-csv manifest.csv

# 3. Train deterministic MeshNet (warm-start)
python 02_train_meshnet.py --manifest manifest.csv --epochs 50

# 4. Train Bayesian MeshNet
python 03_train_bayesian.py --manifest manifest.csv --warmstart checkpoints/meshnet/model.pth

# 5. Evaluate
python 04_evaluate.py --model checkpoints/bayesian/model.pth --manifest manifest.csv

# 6. Compare with original kwyk
python 05_compare_kwyk.py --new-model checkpoints/bayesian/model.pth --kwyk-dir ../../kwyk
```

## Configuration

Edit `config.yaml` to change hyperparameters. Key parameters match the
original kwyk study: filters=96, block_shape=32³, lr=0.0001, ELBO loss.

## GPU Requirements

- Smoke test (16³ blocks, filters=16): Any GPU with ≥4GB
- Standard training (32³, filters=96): GPU with ≥16GB (T4, V100)
- Full-brain inference (256³): GPU with ≥24GB (A100, V100-32GB)
