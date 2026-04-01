# Experiment 02: Binary (2-class) Bayesian training

## Rationale

50-class parcellation is a hard problem. Binary brain extraction (brain vs background)
is much simpler and was used in the original smoke tests. If Bayesian models can learn
binary segmentation, the issue is with label complexity, not the Bayesian architecture.

## Plan

1. Use 5 subjects, binary label mapping, 20 epochs
2. Train MeshNet → warm-start bwn_multi (simplest Bayesian variant)
3. Evaluate both mc=True and mc=False
4. If binary works, the 50-class zero Dice is likely a capacity/epochs issue

## Tasks

- [ ] Create binary config
- [ ] Train MeshNet (binary, 5 subjects, 20 epochs)
- [ ] Train bwn_multi (binary, warm-start, 20 epochs)
- [ ] Evaluate both modes
