#!/bin/bash
#SBATCH --job-name=exp02-eval
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-eval-%j.out
#SBATCH --error=slurm-eval-%j.err
set -euo pipefail

cd /orcd/scratch/orcd/013/satra/kwyk_reproduction
source /orcd/data/satra/002/projects/nobrainer/venvs/nobrainer/bin/activate

echo "=== Experiment 02: Binary eval ==="
python experiments/02_20260330_binary_bayesian/eval_binary.py
echo "=== Done: $(date) ==="
