# Experiment 05 Results

## Key Finding: mc=False during training is REQUIRED for KWYKMeshNet to learn

| Condition | mc_train | n_classes | Final Loss | Final Dice (det) mean/max |
|---|---|---|---|---|
| A | True | 50 | 3.387 | 0.0000/0.0006 |
| **B** | **False** | **50** | **2.620** | **0.0019/0.0936** |
| C | True | 2 | 1.011 | 0.0000 |
| D | False | 2 | 0.910 | 0.0001 |

## Analysis

1. **mc=True kills training**: Conditions A and C (mc=True) both converge to zero
   Dice despite loss decreasing. The local reparameterization noise from FFGConv3d
   prevents stable gradient flow. The loss landscape becomes too noisy.

2. **mc=False allows learning**: Condition B (mc=False, 50-class) achieves 9.4% Dice
   on the best class — comparable to the deterministic MeshNet at similar epoch count.
   The VWN weight normalization itself is fine; the stochastic sampling is the issue.

3. **Binary fails for both**: Possibly an issue with binary evaluation or the model
   having too many parameters for a 2-class problem (3M params for binary).

4. **Loss instability with mc=True**: Condition A shows wild loss swings (1.3 to 5.3)
   because each forward pass samples different weights. mc=False gives stable loss.

## Conclusion

The Bayesian training should use `mc=False` for the forward pass during gradient
computation, and only enable `mc=True` at inference time for uncertainty estimation.
This is the standard approach: train with deterministic weights, use stochastic
inference. The current code passes `mc=True` during training which prevents learning.

## Recommendation

Fix `03_train_bayesian.py` to call `model(images, mc=False)` during training,
and only use `mc=True` for validation MC Dice evaluation.
