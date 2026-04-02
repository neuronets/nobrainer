# Experiment 03 Results

## Key Finding: Warm-start transfer bug — sorted key ordering mismatch

**MeshNet Dice (trained):** mean=0.0132, max=0.4738
**KWYKMeshNet (warm-start, mc=False):** mean=0.0006, max=0.0101
**KWYKMeshNet (warm-start, mc=True):** mean=0.0006, max=0.0103

Only 5 of 7+1 layers transferred successfully.

## Root Cause

`warmstart_kwyk_from_deterministic()` sorts state dict keys alphabetically:
```python
det_convs = [(k, v) for k, v in sorted(state.items()) if "weight" in k and v.ndim == 5]
```

This produces ordering: `classifier.weight, encoder.0, encoder.1, ...`

But KWYKMeshNet FFGConv3d layers are: `layer_0, layer_1, ...` (no classifier — it's a regular Conv3d)

So `classifier.weight [50,96,1,1,1]` pairs with `layer_0.conv [96,1,3,3,3]` → shape mismatch!
Then `encoder.0 [96,1,3,3,3]` pairs with `layer_1.conv [96,96,3,3,3]` → shape mismatch!
Then `encoder.1 [96,96,3,3,3]` pairs with `layer_2.conv [96,96,3,3,3]` → OK (but wrong weights!)

Result: 5 layers "transferred" but with wrong weight assignments (encoder.1→layer_2 instead of
encoder.0→layer_0), and first two layers get random initialization.

## Fix

Filter out the classifier weight before pairing, OR use explicit name matching.

## Conclusion

The Bayesian zero Dice is caused by a broken warm-start. The model starts from
mostly-random weights and 20 epochs isn't enough to learn from scratch.
