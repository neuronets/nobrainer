# Experiment 05: KWYKMeshNet from scratch (no warm-start)

## Rationale

Experiments 03-04 showed warm-start doesn't transfer well. But the Bayesian
training also shows zero Dice after 20 epochs from scratch. The question:
can KWYKMeshNet learn AT ALL with mc=False (deterministic)?

If not, the VWN architecture + CrossEntropyLoss may have a fundamental issue.
We also test: (a) mc=False during training, (b) binary labels for simplicity.

## Plan

1. Train KWYKMeshNet (50-class, mc=True during training, 5 subj, 50 epochs)
2. Train KWYKMeshNet (50-class, mc=FALSE during training, 5 subj, 50 epochs)
3. Train KWYKMeshNet (binary, mc=False, 5 subj, 50 epochs)
4. Compare: does turning off MC during training help? Does binary help?
