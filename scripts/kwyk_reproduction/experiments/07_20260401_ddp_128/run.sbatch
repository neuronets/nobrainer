#!/bin/bash
#SBATCH --job-name=exp07-ddp
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Experiment 07: DDP test on 2× L40S with 128³ patches, 1 epoch
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/07_20260401_ddp_128"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
MANIFEST="$WORK_DIR/kwyk_manifest_500.csv"
ZARR_STORE="$WORK_DIR/data/kwyk_500.zarr"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 07: DDP 2× L40S, 128³, 1 epoch ==="
echo "Node: $(hostname)"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.0f}GB')
"

python 02_train_meshnet.py \
    --manifest "$MANIFEST" \
    --config "$EXP_DIR/config.yaml" \
    --output-dir "$EXP_DIR/checkpoints/meshnet" \
    --epochs 1

echo "=== Done: $(date) ==="
