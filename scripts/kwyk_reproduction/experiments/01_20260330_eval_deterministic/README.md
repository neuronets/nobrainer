# Experiment 01: Evaluate Bayesian models in deterministic mode

## Rationale

All 3 Bayesian variants show zero Dice during MC evaluation despite training loss
decreasing from ~3.8 to ~2.2. The prediction code calls `model(tensor)` which
defaults to `mc=True` in KWYKMeshNet.forward(), activating local reparameterization
noise and dropout. With only 20 epochs of Bayesian training, this noise may
overwhelm the learned signal.

**Hypothesis:** The model weights have learned meaningful representations, but MC
inference noise destroys the output. Evaluating with `mc=False` should show non-zero
Dice.

## Plan

1. Write a quick eval script that loads each Bayesian checkpoint and runs prediction
   with `mc=False` (deterministic forward pass)
2. Compare per-class Dice between mc=True and mc=False
3. No retraining needed — just evaluate existing checkpoints

## Tasks

- [x] Write eval script with mc=False support
- [x] Run on existing 20-epoch checkpoints
- [ ] Compare results
