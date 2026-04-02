#!/bin/bash
#SBATCH --job-name=exp08-full
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue
#SBATCH --signal=USR1@120
#
# DDP 128³ on full KWYK dataset (11,479 subjects, 2× L40S)
# Checkpoints every epoch for resume after preemption
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/08_20260401_ddp_128_full"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="$EXP_DIR/config.yaml"
MANIFEST="$WORK_DIR/kwyk_manifest_full.csv"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 08: DDP 128³ full dataset ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.0f}GB')
"

echo "=== Training MeshNet 128³ (20 epochs, full dataset) ==="
python 02_train_meshnet.py \
    --manifest "$MANIFEST" \
    --config "$CONFIG" \
    --output-dir "$EXP_DIR/checkpoints/meshnet_128" \
    --epochs 20

echo "=== Done: $(date) ==="
