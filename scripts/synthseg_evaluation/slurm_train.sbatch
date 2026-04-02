#!/bin/bash
#SBATCH --job-name=synthseg-eval
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# SynthSeg Evaluation — SLURM Preemptible Training
#
# Usage:
#   sbatch slurm_train.sbatch
#   SYNTHSEG_MODE=mixed SYNTHSEG_MODEL=swin_unetr sbatch slurm_train.sbatch

set -euo pipefail

MODE="${SYNTHSEG_MODE:-real}"
MODEL="${SYNTHSEG_MODEL:-unet}"
CONFIG="${SYNTHSEG_CONFIG:-config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SynthSeg SLURM Training ==="
echo "Model: $MODEL, Mode: $MODE"
echo "Job: ${SLURM_JOB_ID:-local}, Restart: ${SLURM_RESTART_COUNT:-0}"

cd "$SCRIPT_DIR"

python 02_train.py --config "$CONFIG" --mode "$MODE" --model "$MODEL" \
    --manifest manifest.csv
python 03_evaluate.py --model "checkpoints/${MODEL}_${MODE}" \
    --manifest manifest.csv --config "$CONFIG"
