#!/bin/bash
#SBATCH --job-name=zarr-shard
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-zarr-shard-%A_%a.out
#SBATCH --error=slurm-zarr-shard-%A_%a.err
#SBATCH --array=0-114
#
# Job array: each task writes one shard (100 subjects) to the Zarr store.
# 11479 subjects / 100 per shard = 115 shards (0-114)
# Task 0 also creates the store.
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
MANIFEST="$WORK_DIR/kwyk_manifest_full.csv"
ZARR_OUT="$WORK_DIR/data/kwyk_full.zarr"
SUBJECTS_PER_SHARD=100

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== Shard 0: creating store + writing first shard ==="
    python convert_zarr_shard.py \
        --manifest "$MANIFEST" \
        --zarr-store "$ZARR_OUT" \
        --shard-idx 0 \
        --subjects-per-shard "$SUBJECTS_PER_SHARD" \
        --create
else
    # Wait briefly for shard 0 to create the store
    sleep 10
    echo "=== Shard ${SLURM_ARRAY_TASK_ID}: writing ==="
    python convert_zarr_shard.py \
        --manifest "$MANIFEST" \
        --zarr-store "$ZARR_OUT" \
        --shard-idx "$SLURM_ARRAY_TASK_ID" \
        --subjects-per-shard "$SUBJECTS_PER_SHARD"
fi

echo "=== Done: $(date) ==="
