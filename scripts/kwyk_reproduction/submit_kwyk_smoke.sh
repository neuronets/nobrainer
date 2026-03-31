#!/bin/bash
# Submit the full KWYK PAC smoke test pipeline as parallel SLURM jobs:
#
#   Job 1: MeshNet (deterministic warm-start)
#   Jobs 2-4: 3 Bayesian variants (parallel, depend on Job 1)
#   Job 5: Evaluate all variants (depends on Jobs 2-4)
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Submitting KWYK PAC smoke test pipeline ==="

# Step 1: MeshNet
MESHNET_JOB=$(sbatch --parsable slurm_kwyk_smoke.sbatch)
echo "MeshNet:          job ${MESHNET_JOB}"

# Step 2: Bayesian variants (parallel, depend on MeshNet)
BAYES_JOBS=""
for variant in bwn_multi bvwn_multi_prior bayesian_gaussian; do
    JOB=$(sbatch --parsable --dependency=afterok:${MESHNET_JOB} slurm_kwyk_bayesian.sbatch "$variant")
    echo "${variant}:  job ${JOB} (after ${MESHNET_JOB})"
    BAYES_JOBS="${BAYES_JOBS:+${BAYES_JOBS},}${JOB}"
done

# Step 3: Evaluate (depends on all Bayesian jobs + MeshNet)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${MESHNET_JOB}:${BAYES_JOBS} slurm_kwyk_evaluate.sbatch)
echo "Evaluate:         job ${EVAL_JOB} (after all training)"

echo ""
echo "=== Pipeline submitted ==="
echo "Monitor: squeue -u \$USER"
