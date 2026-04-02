# Experiment 01 Results

## Finding

**Both deterministic and MC modes produce zero Dice for all 3 Bayesian variants.**

| Variant | Deterministic | MC |
|---|---|---|
| bwn_multi | 0.0000 | 0.0000 |
| bvwn_multi_prior | 0.0000 | 0.0001 |
| bayesian_gaussian | 0.0000 | 0.0000 |

## Conclusion

The hypothesis was wrong — the issue is NOT MC noise destroying learned signals.
The weights themselves contain no useful information. Possible causes:

1. **Warm-start transfer failure**: weights not properly transferred from MeshNet to KWYKMeshNet
2. **Bayesian training destroying warm-start**: ELBO loss / weight perturbation undoing the transferred weights
3. **Architecture mismatch**: MeshNet vs KWYKMeshNet parameter shapes may not align

## Next Steps

- Investigate warm-start transfer code
- Check if Bayesian model immediately after warm-start (before training) can segment
- Compare model architectures between MeshNet and KWYKMeshNet
