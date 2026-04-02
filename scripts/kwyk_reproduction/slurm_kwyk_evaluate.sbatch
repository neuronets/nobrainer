#!/bin/bash
#SBATCH --job-name=kwyk-eval
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-kwyk-eval-%j.out
#SBATCH --error=slurm-kwyk-eval-%j.err
#
# KWYK Smoke Test — Evaluate all variants (runs after all training completes)
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
CONFIG="config_kwyk_smoke.yaml"
MANIFEST="kwyk_manifest_500.csv"

echo "=== KWYK Evaluation ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

for v in kwyk_smoke_meshnet kwyk_smoke_bwn_multi kwyk_smoke_bvwn_multi_prior kwyk_smoke_bayesian_gaussian; do
    if [ -f "checkpoints/$v/model.pth" ]; then
        echo "--- Evaluating $v (deterministic) ---"
        python 04_evaluate.py --model "checkpoints/$v" \
            --manifest "$MANIFEST" --config "$CONFIG" \
            --split test --n-samples 0 \
            --output-dir "results/${v}_det"
        echo "--- Evaluating $v (MC, 3 samples) ---"
        python 04_evaluate.py --model "checkpoints/$v" \
            --manifest "$MANIFEST" --config "$CONFIG" \
            --split test --n-samples 3 \
            --output-dir "results/${v}_mc"
    else
        echo "WARN: checkpoints/$v/model.pth not found, skipping"
    fi
done

echo "=== Evaluation complete: $(date) ==="
echo "Results:"
for csv in results/kwyk_smoke_*/dice_scores.csv; do
    [ -f "$csv" ] && echo "  $csv: $(tail -n +2 "$csv" | wc -l) volumes"
done
