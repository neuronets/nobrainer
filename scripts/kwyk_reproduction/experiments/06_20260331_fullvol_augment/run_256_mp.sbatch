#!/bin/bash
#SBATCH --job-name=exp06-mp
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-256mp-%j.out
#SBATCH --error=slurm-256mp-%j.err
#
# Experiment 06d: Full 256³ + model parallelism on 2× L40S
# Layers split across GPUs — each GPU holds ~half the model
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/06_20260331_fullvol_augment"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="$EXP_DIR/config_256_mp.yaml"
MANIFEST="$WORK_DIR/kwyk_manifest_500.csv"
ZARR_STORE="$WORK_DIR/data/kwyk_500.zarr"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 06d: 256³ + model parallel (2× L40S) ==="
echo "Node: $(hostname)"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.0f}GB')
"

# MeshNet 256³ with model parallelism
echo "=== Step 1: MeshNet 256³ (model parallel) ==="
python 02_train_meshnet.py --manifest "$MANIFEST" --config "$CONFIG" \
    --output-dir "$EXP_DIR/checkpoints/meshnet_256mp" --epochs 20

# bwn_multi 256³ with model parallelism
echo "=== Step 2: bwn_multi 256³ (model parallel) ==="
python 03_train_bayesian.py --manifest "$MANIFEST" --config "$CONFIG" \
    --variant bwn_multi --warmstart "$EXP_DIR/checkpoints/meshnet_256mp" \
    --output-dir "$EXP_DIR/checkpoints/bwn_multi_256mp" --epochs 20

# Evaluate
echo "=== Step 3: Evaluate ==="
for v in meshnet_256mp bwn_multi_256mp; do
    if [ -f "$EXP_DIR/checkpoints/$v/model.pth" ]; then
        echo "--- $v (deterministic) ---"
        python 04_evaluate.py --model "$EXP_DIR/checkpoints/$v" \
            --manifest "$MANIFEST" --config "$CONFIG" \
            --split test --n-samples 0 \
            --output-dir "$EXP_DIR/results/${v}_det"
    fi
done

echo "=== Done: $(date) ==="
