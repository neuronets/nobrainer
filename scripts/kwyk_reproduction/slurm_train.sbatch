#!/bin/bash
#SBATCH --job-name=kwyk-train
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
# KWYK Brain Extraction Reproduction — SLURM Preemptible Training
#
# Trains all kwyk model variants with automatic checkpoint/resume on
# preemption and automatic batch size optimization per GPU.
#
# Usage:
#   sbatch slurm_train.sbatch                             # 1 GPU
#   sbatch --gres=gpu:4 slurm_train.sbatch                # 4 GPUs
#   sbatch --partition=gpu slurm_train.sbatch              # Non-preemptible
#   KWYK_EPOCHS=100 sbatch slurm_train.sbatch              # More epochs
#
# Multi-GPU:
#   Batch size is auto-optimized per GPU via nobrainer.gpu.auto_batch_size.
#   Request more GPUs with --gres=gpu:N.
#
# Environment variables:
#   KWYK_DATASETS    — space-separated OpenNeuro IDs (default: ds000114)
#   KWYK_EPOCHS      — epochs per variant (default: 50)
#   KWYK_WORK_DIR    — working directory (default: $SLURM_SUBMIT_DIR)
#   KWYK_VENV        — path to venv (default: .venv-kwyk)

set -euo pipefail

WORK_DIR="${KWYK_WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV_DIR="${KWYK_VENV:-${WORK_DIR}/.venv-kwyk}"
DATASETS="${KWYK_DATASETS:-ds000114}"
EPOCHS="${KWYK_EPOCHS:-50}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOBRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== KWYK SLURM Training ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-unknown}"
echo "Restart:    ${SLURM_RESTART_COUNT:-0}"

cd "$WORK_DIR"

# --- Setup venv (first run only) ---
if [ ! -d "$VENV_DIR" ]; then
    uv venv --python 3.14 "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
uv pip install -e "${NOBRAINER_ROOT}[bayesian,versioning,dev]" \
    monai pyro-ppl datalad matplotlib pyyaml scipy nibabel 2>&1 | tail -3
uv tool install git-annex 2>/dev/null || true

# --- Show GPU info ---
python -c "
from nobrainer.gpu import gpu_info, gpu_count
print('GPUs:', gpu_count())
for g in gpu_info():
    print('  GPU {id}: {name} ({memory_gb} GB)'.format(**g))
"

cd "$SCRIPT_DIR"

# --- Step 1: Data ---
if [ ! -f manifest.csv ]; then
    # shellcheck disable=SC2086
    python 01_assemble_dataset.py --datasets $DATASETS \
        --output-csv manifest.csv --output-dir data --label-mapping binary
fi

# --- Step 2: Deterministic MeshNet ---
python 02_train_meshnet.py --manifest manifest.csv --config config.yaml \
    --output-dir checkpoints/meshnet --epochs "$EPOCHS"

# --- Step 3: All Bayesian variants (auto batch size, checkpoint/resume) ---
for variant in bwn_multi bvwn_multi_prior; do
    echo "=== Training $variant ==="
    python 03_train_bayesian.py --manifest manifest.csv --config config.yaml \
        --variant "$variant" --warmstart checkpoints/meshnet \
        --output-dir "checkpoints/$variant" --epochs "$EPOCHS"
done

# --- Step 4: Evaluate ---
for v in checkpoints/meshnet checkpoints/bwn_multi checkpoints/bvwn_multi_prior; do
    [ -f "$v/model.pth" ] && python 04_evaluate.py --model "$v/model.pth" \
        --manifest manifest.csv --split test --n-samples 10 \
        --output-dir "results/$(basename $v)" || true
done

echo "=== Done: $(ls checkpoints/*/model.pth 2>/dev/null | wc -l) models ==="
