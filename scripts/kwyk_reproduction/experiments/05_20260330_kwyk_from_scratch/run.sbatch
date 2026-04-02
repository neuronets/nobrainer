#!/bin/bash
#SBATCH --job-name=exp05-scratch
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

cd /orcd/scratch/orcd/013/satra/kwyk_reproduction
source /orcd/data/satra/002/projects/nobrainer/venvs/nobrainer/bin/activate

echo "=== Experiment 05: KWYKMeshNet from scratch ==="
python experiments/05_20260330_kwyk_from_scratch/run.py
echo "=== Done: $(date) ==="
