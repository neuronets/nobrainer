#!/bin/bash
#SBATCH --job-name=exp06-256
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-256-%j.out
#SBATCH --error=slurm-256-%j.err
#
# Experiment 06b: Full 256³ volume + augmentation on H200 (141GB)
# Single H200 fits 256³ with batch=1 (~90GB forward+backward)
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/06_20260331_fullvol_augment"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="$EXP_DIR/config_256.yaml"
MANIFEST="$WORK_DIR/kwyk_manifest_500.csv"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 06b: Full 256³ + augmentation (2× A100) ==="
echo "Node: $(hostname)"
python -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.0f}GB')
"

# MeshNet 256³ (20 epochs)
echo "=== Step 1: MeshNet 256³ ==="
python 02_train_meshnet.py --manifest "$MANIFEST" --config "$CONFIG" \
    --output-dir "$EXP_DIR/checkpoints/meshnet_256" --epochs 20

# bwn_multi 256³ (20 epochs, warm-start)
echo "=== Step 2: bwn_multi 256³ ==="
python 03_train_bayesian.py --manifest "$MANIFEST" --config "$CONFIG" \
    --variant bwn_multi --warmstart "$EXP_DIR/checkpoints/meshnet_256" \
    --output-dir "$EXP_DIR/checkpoints/bwn_multi_256" --epochs 20

# Evaluate
echo "=== Step 3: Evaluate ==="
for v in meshnet_256 bwn_multi_256; do
    if [ -f "$EXP_DIR/checkpoints/$v/model.pth" ]; then
        echo "--- $v (deterministic) ---"
        python 04_evaluate.py --model "$EXP_DIR/checkpoints/$v" \
            --manifest "$MANIFEST" --config "$CONFIG" \
            --split test --n-samples 0 \
            --output-dir "$EXP_DIR/results/${v}_det"
    fi
done

echo "=== Done: $(date) ==="
