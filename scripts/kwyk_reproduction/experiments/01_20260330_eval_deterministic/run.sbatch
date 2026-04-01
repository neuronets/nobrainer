#!/bin/bash
#SBATCH --job-name=exp01-det
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

cd /orcd/scratch/orcd/013/satra/kwyk_reproduction
source /orcd/data/satra/002/projects/nobrainer/venvs/nobrainer/bin/activate

echo "=== Experiment 01: Deterministic eval of Bayesian models ==="
echo "Started: $(date)"

python experiments/01_20260330_eval_deterministic/eval_deterministic.py

echo "=== Done: $(date) ==="
