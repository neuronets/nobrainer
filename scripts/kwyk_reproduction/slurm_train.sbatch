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
# This script trains all 3 kwyk model variants on a preemptible partition.
# When the job is preempted, SLURM sends SIGUSR1 120 seconds before
# killing the process.  The training scripts checkpoint and exit cleanly,
# then --requeue re-submits the job.  On restart, training resumes from
# the last checkpoint automatically.
#
# Usage:
#   sbatch slurm_train.sbatch                    # Full pipeline
#   sbatch --partition=gpu slurm_train.sbatch     # Non-preemptible
#   sbatch --time=4:00:00 slurm_train.sbatch      # Shorter time limit
#
# Environment variables (override defaults):
#   KWYK_DATASETS    — space-separated OpenNeuro IDs (default: ds000114)
#   KWYK_EPOCHS      — epochs per model variant (default: 50)
#   KWYK_WORK_DIR    — working directory (default: $SLURM_SUBMIT_DIR)
#   KWYK_VENV        — path to Python venv (default: .venv-kwyk)
#
# Requirements:
#   - Python 3.12+ with uv
#   - CUDA-capable GPU
#   - Network access for OpenNeuro (first run only)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORK_DIR="${KWYK_WORK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV_DIR="${KWYK_VENV:-${WORK_DIR}/.venv-kwyk}"
DATASETS="${KWYK_DATASETS:-ds000114}"
EPOCHS="${KWYK_EPOCHS:-50}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOBRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== KWYK SLURM Training ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "Work dir:   ${WORK_DIR}"
echo "Datasets:   ${DATASETS}"
echo "Epochs:     ${EPOCHS}"
echo "Restart:    ${SLURM_RESTART_COUNT:-0}"
echo ""

cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Virtual environment setup (only on first run)
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    uv venv --python 3.14 "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Installing dependencies..."
uv pip install -e "${NOBRAINER_ROOT}[bayesian,versioning,dev]" \
    monai pyro-ppl datalad matplotlib pyyaml scipy nibabel 2>&1 | tail -3
uv tool install git-annex 2>/dev/null || true

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# Step 1: Assemble dataset (skipped if manifest exists)
# ---------------------------------------------------------------------------
cd "$SCRIPT_DIR"

if [ ! -f manifest.csv ]; then
    echo "=== Step 1: Assembling dataset ==="
    # shellcheck disable=SC2086
    python 01_assemble_dataset.py \
        --datasets $DATASETS \
        --output-csv manifest.csv \
        --output-dir data \
        --label-mapping binary
else
    echo "=== Step 1: manifest.csv exists, skipping data assembly ==="
    wc -l manifest.csv
fi

# ---------------------------------------------------------------------------
# Step 2: Train deterministic MeshNet (warm-start foundation)
# ---------------------------------------------------------------------------
echo "=== Step 2: Training deterministic MeshNet ==="
python 02_train_meshnet.py \
    --manifest manifest.csv \
    --config config.yaml \
    --output-dir checkpoints/meshnet \
    --epochs "$EPOCHS"

# ---------------------------------------------------------------------------
# Step 3: Train all Bayesian variants (with checkpoint/resume)
# ---------------------------------------------------------------------------
for variant in bwn_multi bvwn_multi_prior bayesian_gaussian; do
    echo ""
    echo "=== Step 3: Training variant ${variant} ==="
    python 03_train_bayesian.py \
        --manifest manifest.csv \
        --config config.yaml \
        --variant "$variant" \
        --warmstart checkpoints/meshnet \
        --output-dir "checkpoints/${variant}" \
        --epochs "$EPOCHS"
done

# ---------------------------------------------------------------------------
# Step 4: Evaluate all variants
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Evaluating all variants ==="
for variant_dir in checkpoints/meshnet checkpoints/bwn_multi checkpoints/bvwn_multi_prior checkpoints/bayesian_gaussian; do
    variant_name=$(basename "$variant_dir")
    if [ -f "$variant_dir/model.pth" ]; then
        echo "  Evaluating ${variant_name}..."
        python 04_evaluate.py \
            --model "$variant_dir/model.pth" \
            --manifest manifest.csv \
            --split test \
            --n-samples 10 \
            --output-dir "results/${variant_name}" || true
    fi
done

echo ""
echo "=== Training complete ==="
echo "Checkpoints: $(ls checkpoints/*/model.pth 2>/dev/null | wc -l) models"
echo "Results:     $(ls results/*/dice_scores.csv 2>/dev/null | wc -l) evaluations"
ls -la checkpoints/*/ 2>/dev/null || true
