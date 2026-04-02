# Experiment 04: Fix warm-start and verify Bayesian learning

## Rationale

Experiment 03 found that `warmstart_kwyk_from_deterministic()` has a key ordering
bug: sorted() puts classifier.weight before encoder.X, causing all layer pairings
to be offset. Fix the transfer, verify Dice is preserved, then train Bayesian.

## Plan

1. Fix warm-start: filter classifier from det_convs, transfer it separately
2. Verify fixed warm-start preserves MeshNet Dice
3. Train bwn_multi for 20 epochs with fixed warm-start
4. Evaluate in both mc=False and mc=True modes
5. Use 5 subjects for speed (sanity manifest)

## Tasks

- [ ] Fix warm-start function
- [ ] Verify transfer preserves Dice
- [ ] Train and evaluate Bayesian
