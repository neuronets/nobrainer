# Bayesian Model Learning Experiments — Task Planner

**Session dates:** 2026-03-30 to 2026-03-31

## Root Causes Found

### 1. Warm-start key ordering bug (Exp 03)
`sorted(state.items())` puts `classifier.weight` before `encoder.X`.
**Fixed:** filter classifier, transfer separately. Branch: `fix/warmstart-key-ordering`.

### 2. Single mc= flag (Exp 05 + original TF code review)
Original TF bwn trains with `is_mc_v=False` (deterministic VWN) + `is_mc_b=True`
(bernoulli dropout ON). PyTorch had one `mc=` flag controlling both.
**Fixed:** `mc_vwn` and `mc_dropout` independent flags. Branch: `fix/kwyk-decouple-mc-flags`.

### 3. Dropout ordering mismatch
Original TF: conv → dropout → relu. PyTorch had: conv → relu → dropout.
**Fixed** in same branch.

### 4. Data pipeline bottleneck
NIfTI streaming (20K reads/epoch) too slow. Zarr3 with sharding (1 file per array,
32³ chunk-aligned reads) eliminates this.
**Fixed:** sharded Zarr3 conversion + PatchDataset zarr:// support.

### 5. auto_batch_size profiling with wrong mode
Profiled with mc=True (both VWN+dropout) but training uses mc_vwn=False.
**Fixed:** forward_kwargs parameter in auto_batch_size.

## Additional Discrepancies Found (from original TF code review)

| Aspect | Original TF | Current PyTorch |
|---|---|---|
| Framework | TensorFlow 1.12 | PyTorch 2.11 |
| Dropout rate (bwn) | keep_prob=0.5 (50%) | dropout_rate=0.25 |
| Dropout type | tf.nn.dropout (element-wise, no rescale) | nn.Dropout3d (spatial, rescales) |
| Loss regularization | L2 weight decay: sum(mu²)/(2*N) | ELBOLoss with KL=0 (just CE) |
| sigma_prior (bwn) | 1.0 | 0.1 |
| Classifier layer | VWN conv3d | Standard nn.Conv3d |
| prior_path support | Yes (load from trained model) | Missing |
| Concrete dropout p_prior | 0.5 | 0.9 |
| Subjects | ~10,000 | 500 (current) |

## Experiment Log

| # | Name | Status | Key Finding |
|---|---|---|---|
| 01 | Eval det mode | DONE | Zero Dice both modes — weights empty |
| 02 | Binary Bayesian | DONE | Zero Dice — same issue |
| 03 | Warm-start diagnostic | DONE | **BUG: sorted key ordering** |
| 04 | Fixed warm-start | DONE | Transfer improved but training destroys signal |
| 05 | From scratch | DONE | **mc=False trains, mc=True doesn't** |
| 06 | Original TF code review | DONE | **is_mc_v=False, is_mc_b=True in original** |

## Current Pipeline (running)

- Job 11222646: Zarr3 conversion (500 subjects, sharded)
- Jobs 11222647-49: Bayesian training (mc_vwn=False, mc_dropout=True, streaming Zarr)
- Job 11222650: Evaluation (det + MC)

## Next Steps

1. Match dropout rate to original (0.5 not 0.25)
2. Add L2 weight decay to match original loss (not KL)
3. Scale to more subjects / epochs
4. Implement prior_path for multi-stage training
