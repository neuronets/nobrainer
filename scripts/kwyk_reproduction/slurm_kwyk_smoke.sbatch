#!/bin/bash
#SBATCH --job-name=kwyk-pac-smoke
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-kwyk-smoke-%j.out
#SBATCH --error=slurm-kwyk-smoke-%j.err
#
# KWYK Training — Step 1: deterministic MeshNet + manifest build
# Steps 2-4 (Bayesian variants) are launched as dependent parallel jobs
# by submit_kwyk_smoke.sh
# Set KWYK_EPOCHS env var to override (default: 20)
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
DATA_DIR="/orcd/scratch/orcd/013/satra/data/SharedData/segmentation/freesurfer_asegs"
CONFIG="config_kwyk_smoke.yaml"
MANIFEST="kwyk_manifest_500.csv"
N_SUBJECTS=500

echo "=== KWYK PAC Dataset Smoke Test — MeshNet ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# --- Build manifest if needed ---
if [ ! -f "$MANIFEST" ]; then
    echo "=== Building manifest (${N_SUBJECTS} subjects) ==="
    python build_kwyk_manifest.py \
        --data-dir "$DATA_DIR" \
        --output-csv "$MANIFEST" \
        --n-subjects "$N_SUBJECTS" \
        --seed 42
fi

echo "Split counts:"
tail -n +2 "$MANIFEST" | cut -d, -f5 | sort | uniq -c

EPOCHS="${KWYK_EPOCHS:-20}"

# --- Train deterministic MeshNet ---
echo "=== Training deterministic MeshNet (${EPOCHS} epochs) ==="
python 02_train_meshnet.py --manifest "$MANIFEST" --config "$CONFIG" \
    --output-dir checkpoints/kwyk_smoke_meshnet --epochs "$EPOCHS"

echo "=== MeshNet complete: $(date) ==="
