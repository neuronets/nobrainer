#!/bin/bash
#SBATCH --job-name=kwyk-bayes
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-kwyk-bayes-%j.out
#SBATCH --error=slurm-kwyk-bayes-%j.err
#
# KWYK Smoke Test — Train one Bayesian variant (launched in parallel)
#
# Usage (via submit_kwyk_smoke.sh, not directly):
#   sbatch --dependency=afterok:$MESHNET_JOB slurm_kwyk_bayesian.sbatch bwn_multi
#
set -euo pipefail

VARIANT="${1:?Usage: sbatch slurm_kwyk_bayesian.sbatch <variant>}"
WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="config_kwyk_smoke.yaml"
MANIFEST="kwyk_manifest_500.csv"

echo "=== KWYK Bayesian Training: ${VARIANT} ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

EPOCHS="${KWYK_EPOCHS:-20}"

echo "=== Training ${VARIANT} (${EPOCHS} epochs) ==="
python 03_train_bayesian.py --manifest "$MANIFEST" --config "$CONFIG" \
    --variant "$VARIANT" --warmstart checkpoints/kwyk_smoke_meshnet \
    --output-dir "checkpoints/kwyk_smoke_${VARIANT}" --epochs "$EPOCHS"

echo "=== ${VARIANT} complete: $(date) ==="
