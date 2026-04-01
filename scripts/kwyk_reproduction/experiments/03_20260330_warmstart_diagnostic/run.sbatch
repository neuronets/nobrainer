#!/bin/bash
#SBATCH --job-name=exp03-ws
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

cd /orcd/scratch/orcd/013/satra/kwyk_reproduction
source /orcd/data/satra/002/projects/nobrainer/venvs/nobrainer/bin/activate

echo "=== Experiment 03: Warm-start diagnostic ==="
python experiments/03_20260330_warmstart_diagnostic/diagnose.py
echo "=== Done: $(date) ==="
