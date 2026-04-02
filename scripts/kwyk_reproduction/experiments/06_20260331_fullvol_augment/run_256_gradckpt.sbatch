#!/bin/bash
#SBATCH --job-name=exp06-gc
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-256gc-%j.out
#SBATCH --error=slurm-256gc-%j.err
#
# Experiment 06c: Full 256³ + gradient checkpointing on single A100 (80GB)
# Gradient checkpointing halves activation memory: ~90GB → ~45GB
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/06_20260331_fullvol_augment"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="$EXP_DIR/config_256.yaml"
MANIFEST="$WORK_DIR/kwyk_manifest_500.csv"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 06c: 256³ + gradient checkpointing (A100) ==="
echo "Node: $(hostname)"
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')
"

# MeshNet 256³ with gradient checkpointing
echo "=== Step 1: MeshNet 256³ (gradient checkpointing) ==="
python -c "
import sys; sys.path.insert(0, '.')
from nobrainer.processing.dataset import Dataset
from nobrainer.processing.segmentation import Segmentation
from utils import load_config, setup_logging
import torch

log = setup_logging('exp06c')
config = load_config('$CONFIG')
block_shape = tuple(config['block_shape'])
n_classes = config['n_classes']
label_mapping = config.get('label_mapping', 'binary')

# Load data
import csv
train_pairs, val_pairs = [], []
with open('$MANIFEST') as f:
    for row in csv.DictReader(f):
        p = (row['t1w_path'], row['label_path'])
        if row['split'] == 'train': train_pairs.append(p)
        elif row['split'] == 'val': val_pairs.append(p)

ds_train = Dataset.from_files(train_pairs, block_shape=block_shape, n_classes=n_classes).batch(1).binarize(label_mapping).streaming(patches_per_volume=1)
ds_val = Dataset.from_files(val_pairs[:5], block_shape=block_shape, n_classes=n_classes).batch(1).binarize(label_mapping).streaming(patches_per_volume=1)

seg = Segmentation(base_model='meshnet', model_args={'n_classes': n_classes, 'filters': 96, 'receptive_field': 37, 'dropout_rate': 0.25}, checkpoint_filepath='$EXP_DIR/checkpoints/meshnet_256gc')

def _log(epoch, logs, model):
    msg = f'Epoch {epoch+1}/20: loss={logs[\"loss\"]:.4f}'
    if 'val_loss' in logs: msg += f' val_loss={logs[\"val_loss\"]:.4f} val_acc={logs[\"val_acc\"]:.4f}'
    log.info(msg)

seg.fit(dataset_train=ds_train, dataset_validate=ds_val, epochs=20, optimizer=torch.optim.Adam, opt_args={'lr': 0.0001}, callbacks=[_log], checkpoint_freq=5, gradient_checkpointing=True)
log.info('Done. Result: %s', seg._training_result)
"

echo "=== Done: $(date) ==="
