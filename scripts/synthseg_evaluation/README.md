# SynthSeg Evaluation Pipeline

Evaluate SynthSeg-based training against real-data baselines using
multiple model architectures.

## Quick Start

```bash
cd scripts/synthseg_evaluation

# Smoke test (2 epochs, unet, real+mixed)
./run.sh --smoke-test

# Full evaluation (all models × all modes from config.yaml)
./run.sh
```

## Training Modes

| Mode | Description |
|------|-------------|
| `real` | Train on real data only (baseline) |
| `synthetic` | Train on SynthSeg-generated data only |
| `mixed` | Train on mix of real + synthetic (configurable ratio) |

## Available Models

| Model | Architecture | Source |
|-------|-------------|--------|
| `unet` | 3D U-Net | MONAI |
| `swin_unetr` | Swin Transformer U-Net | MONAI |
| `segresnet` | Residual Encoder SegNet | MONAI |
| `kwyk_meshnet` | VWN MeshNet + dropout | nobrainer |
| `attention_unet` | Attention U-Net | MONAI |

## Configuration

Edit `config.yaml` to change models, training modes, SynthSeg parameters,
and data settings. Key options:

- `training.modes`: which modes to evaluate
- `training.mixed_ratio`: fraction of synthetic data in mixed mode
- `models`: list of model architectures to compare
- `synthseg.*`: SynthSeg generation parameters

## SLURM

```bash
# Single model+mode
SYNTHSEG_MODE=mixed SYNTHSEG_MODEL=swin_unetr sbatch slurm_train.sbatch

# All combinations
for model in unet swin_unetr kwyk_meshnet; do
  for mode in real synthetic mixed; do
    SYNTHSEG_MODE=$mode SYNTHSEG_MODEL=$model sbatch slurm_train.sbatch
  done
done
```

## Output

```
results/
├── comparison_table.csv     # Dice per model × mode
└── comparison_figure.png    # Bar chart visualization
checkpoints/
├── unet_real/eval/          # Per-model eval results
├── unet_mixed/eval/
├── swin_unetr_real/eval/
└── ...
```
