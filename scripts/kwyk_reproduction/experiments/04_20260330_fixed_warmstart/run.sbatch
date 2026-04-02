#!/bin/bash
#SBATCH --job-name=exp04-fix
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

echo "=== Experiment 04: Fixed warm-start + Bayesian training ==="
python experiments/04_20260330_fixed_warmstart/run.py
echo "=== Done: $(date) ==="
