# Experiment 06: Full-volume (256³) training with augmentation

## Rationale

Current training uses 32³ patches — the model never sees global context.
Training on full 256³ volumes should improve segmentation quality,
especially for large structures. Combined with augmentation (affine + flip
+ noise) for regularization.

## Plan

1. Use 128³ blocks on L40S (batch_size=4 fits in 47GB)
   OR request H200/A100 for full 256³ (batch_size=1 per GPU, 2 GPUs)
2. Standard augmentation profile (affine rotation/scale, flips, Gaussian noise)
3. MeshNet first (deterministic baseline), then bwn_multi
4. 20 epochs on 500 subjects
5. Compare Dice vs 32³ patch training

## GPU Options

| GPU | Memory | 256³ batch=1 | 128³ batch=4 |
|-----|--------|-------------|-------------|
| L40S (47GB) | 47GB | OOM (90GB) | OK (11GB) |
| A100 (80GB) | 80GB | Tight | OK |
| H200 (141GB) | 141GB | OK | OK |
| 2× L40S | 94GB | OK | OK |
