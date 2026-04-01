#!/bin/bash
#SBATCH --job-name=exp06-128
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-128-%j.out
#SBATCH --error=slurm-128-%j.err
#
# Experiment 06a: 128³ blocks + augmentation on L40S
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/06_20260331_fullvol_augment"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="$EXP_DIR/config_fullvol.yaml"
MANIFEST="$WORK_DIR/kwyk_manifest_500.csv"
ZARR_STORE="$WORK_DIR/data/kwyk_500.zarr"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 06a: 128³ + augmentation ==="
echo "Node: $(hostname)"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')"

# MeshNet 128³ (20 epochs)
echo "=== Step 1: MeshNet 128³ ==="
python 02_train_meshnet.py --manifest "$MANIFEST" --config "$CONFIG" \
    --output-dir "$EXP_DIR/checkpoints/meshnet_128" --epochs 20

# bwn_multi 128³ (20 epochs, warm-start)
echo "=== Step 2: bwn_multi 128³ ==="
python 03_train_bayesian.py --manifest "$MANIFEST" --config "$CONFIG" \
    --variant bwn_multi --warmstart "$EXP_DIR/checkpoints/meshnet_128" \
    --output-dir "$EXP_DIR/checkpoints/bwn_multi_128" --epochs 20

# Evaluate
echo "=== Step 3: Evaluate ==="
for v in meshnet_128 bwn_multi_128; do
    if [ -f "$EXP_DIR/checkpoints/$v/model.pth" ]; then
        echo "--- $v (deterministic) ---"
        python 04_evaluate.py --model "$EXP_DIR/checkpoints/$v" \
            --manifest "$MANIFEST" --config "$CONFIG" \
            --split test --n-samples 0 \
            --output-dir "$EXP_DIR/results/${v}_det"
    fi
done

echo "=== Done: $(date) ==="
