# Experiment 03: Warm-start transfer diagnostic

## Rationale

Experiment 01 showed that Bayesian model weights have zero Dice even in deterministic
mode. This means the warm-start transfer from MeshNet to KWYKMeshNet may not be
working, OR the Bayesian training loop destroys transferred weights.

## Plan

1. Load trained MeshNet checkpoint
2. Create KWYKMeshNet and run warm-start transfer
3. Evaluate KWYKMeshNet immediately BEFORE any Bayesian training (mc=False)
4. If Dice > 0: warm-start works, Bayesian training is the problem
5. If Dice = 0: warm-start transfer is broken
6. Also compare parameter counts and shapes between MeshNet and KWYKMeshNet

## Tasks

- [ ] Compare architectures
- [ ] Evaluate warm-started model before training
- [ ] Check transfer log messages
