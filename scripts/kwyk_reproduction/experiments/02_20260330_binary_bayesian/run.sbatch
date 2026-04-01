#!/bin/bash
#SBATCH --job-name=exp02-bin
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
EXP_DIR="$WORK_DIR/experiments/02_20260330_binary_bayesian"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
MANIFEST="$WORK_DIR/kwyk_sanity_manifest.csv"  # 5 subjects
CONFIG="$EXP_DIR/config.yaml"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Experiment 02: Binary Bayesian (5 subjects, 20 epochs) ==="
echo "Started: $(date)"

# Train MeshNet (binary)
echo "=== Step 1: MeshNet (binary, 20 epochs) ==="
python 02_train_meshnet.py --manifest "$MANIFEST" --config "$CONFIG" \
    --output-dir "$EXP_DIR/checkpoints/meshnet" --epochs 20

# Train bwn_multi (binary, warm-start from MeshNet)
echo "=== Step 2: bwn_multi (binary, 20 epochs) ==="
python 03_train_bayesian.py --manifest "$MANIFEST" --config "$CONFIG" \
    --variant bwn_multi --warmstart "$EXP_DIR/checkpoints/meshnet" \
    --output-dir "$EXP_DIR/checkpoints/bwn_multi" --epochs 20

# Quick eval: deterministic and MC
echo "=== Step 3: Eval (deterministic + MC) ==="
python experiments/02_20260330_binary_bayesian/eval_binary.py

echo "=== Done: $(date) ==="
