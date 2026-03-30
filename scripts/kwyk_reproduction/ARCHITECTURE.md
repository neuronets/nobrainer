# KWYK Architecture Verification

This document records how the original kwyk model architecture was verified
against the paper, source code, and trained model weights.

## Paper Reference

McClure P. et al., "Knowing What You Know in Brain Segmentation Using
Bayesian Deep Neural Networks", Front. Neuroinform. 2019.
https://doi.org/10.3389/fninf.2019.00067

## Three Model Variants

| Variant | kwyk ID | Conv Layer | Dropout | MC at inference |
|---------|---------|-----------|---------|-----------------|
| MAP | `bwn` (all_50_wn) | VWN | Bernoulli (fixed) | No |
| MC Bernoulli Dropout (BD) | `bwn_multi` (all_50_bwn_09_multi) | VWN | Bernoulli (fixed) | Yes |
| Spike-and-Slab Dropout (SSD) | `bvwn_multi_prior` (all_50_bvwn_multi_prior) | VWN | Concrete (learned) | Yes |

## Architecture: Variational Weight Normalization (VWN) Conv

### Verified from trained model variables

Downloaded `neuronets/kwyk:latest-cpu` Docker container and inspected the
SavedModel variables for the SSD model (`all_50_bvwn_multi_prior`):

```
layer_1/conv3d/v:0: [3, 3, 3, 1, 96]      # raw weight for WN
layer_1/conv3d/g:0: [1, 1, 1, 1, 96]      # gain per filter
layer_1/conv3d/kernel_a:0: [3, 3, 3, 1, 96]  # sigma = |kernel_a|
layer_1/conv3d/bias_m:0: [96]              # bias mean
layer_1/conv3d/bias_a:0: [96]              # bias sigma = |bias_a|
layer_1/concrete_dropout/p:0: [96]         # per-filter dropout rate
```

This confirms **weight normalization** (`v`, `g`) is used, not the direct
`μ` parameterization described in the paper's equations.

### Key finding: all 3 models are independently trained VWN models

All 3 saved models have the **same layer structure** (`v`, `g`, `kernel_a`,
`bias_m`, `bias_a`) — including the MAP model (`all_50_wn`).  They are
**not** weight-sharing variants; they were trained independently:

| Model | Total variables | Extra per layer | Timestamp |
|-------|----------------|----------------|-----------|
| all_50_wn (MAP) | 41 | — | 1555341859 |
| all_50_bwn_09_multi (BD) | 41 | — | 1555963478 |
| all_50_bvwn_multi_prior (SSD) | 48 | `concrete_dropout/p` | 1556816070 |

The MAP and BD models have identical parameterization (both have `kernel_a`
for learned sigma).  The only difference is whether MC sampling is enabled
at inference time.  The SSD model additionally has 7 `concrete_dropout/p`
parameters (one per conv layer) for learned per-filter dropout rates.

### Verified from source code

Commit `4dd379c` in `neuronets/kwyk` repo (Patrick McClure, 2019-02-28):

**`nobrainer/models/vwn_conv.py`** — `_Conv.build()`:
```python
self.v = self.add_variable(name='v', ...)
self.g = self.add_variable(name='g', ...)
self.v_norm = tf.nn.l2_normalize(self.v, [...])
self.kernel_m = tf.multiply(self.g, self.v_norm, name='kernel_m')
self.kernel_a = self.add_variable(name='kernel_a', ...)
self.kernel_sigma = tf.abs(self.kernel_a, name='kernel_sigma')
```

**`_Conv.call()`** — local reparameterization trick:
```python
outputs_mean = self._convolution_op(inputs, self.kernel_m)
outputs_var = self._convolution_op(tf.square(inputs), tf.square(self.kernel_sigma))
outputs_e = tf.random_normal(shape=tf.shape(self.g))
# MC path:
output = outputs_mean + tf.sqrt(outputs_var + 1e-8) * outputs_e
```

**`nobrainer/models/bayesian_dropout.py`** defines:
- `bernoulli_dropout()` — standard MC dropout (bwn/bwn_multi)
- `concrete_dropout()` — learned per-filter rate (bvwn_multi_prior)
- `gaussian_dropout()` — not used in final models

### Paper vs Implementation discrepancy

The paper (Section 2.2.3.2) describes the mean weight as `μ_{f,t}` (Eq. 13),
but the actual implementation uses weight normalization:
- `kernel_m = g · v / ||v||` (Salimans & Kingma 2016)
- This is a reparameterization of the mean that aids training stability
- The sigma is the same in both: `σ_{f,t} = |kernel_a_{f,t}|`

The paper's equations are in terms of the effective mean (`μ`), which is
computed via WN but isn't stored directly as a parameter.

## KL Divergence (Eq. 16-18)

Two terms per filter:

1. **Bernoulli KL** for concrete dropout (Eq. 17):
   `KL(q_p || p_prior) = p·log(p/p_prior) + (1-p)·log((1-p)/(1-p_prior))`
   Prior: `p_prior = 0.5`

2. **Gaussian KL** per weight (Eq. 18):
   `KL(N(μ,σ) || N(μ_prior, σ_prior)) = log(σ_prior/σ) + (σ² + (μ-μ_prior)²)/(2σ²_prior) - 1/2`
   Prior: `μ_prior = 0, σ_prior = 0.1`

## Network Architecture (Table 2)

8 layers of dilated 3×3×3 convolutions:
- Layers 1-3: dilation=1, 96 filters, ReLU
- Layer 4: dilation=2
- Layer 5: dilation=4
- Layer 6: dilation=8
- Layer 7: dilation=1
- Layer 8 (logits): 1×1×1, 50 filters, Softmax

Receptive field = 37 voxels.

## Our Implementation

`nobrainer.models.bayesian.vwn_layers.FFGConv3d`:
- Parameters: `v`, `g` (weight normalization), `kernel_a` (sigma), `bias_m`, `bias_a`
- Forward: local reparameterization trick matching the original
- KL: Eq. 18 with `prior_mu=0, prior_sigma=0.1`

`nobrainer.models.bayesian.vwn_layers.ConcreteDropout3d`:
- Learned `p` per filter via concrete relaxation (Eq. 10)
- KL: Eq. 17 with `prior_p=0.5`

`nobrainer.models.bayesian.kwyk_meshnet.KWYKMeshNet`:
- Registered as `"kwyk_meshnet"` in model registry (no Pyro dependency)
- `dropout_type="bernoulli"` for bwn/bwn_multi
- `dropout_type="concrete"` for bvwn_multi_prior (SSD)
- `mc=True/False` flag controls stochastic vs deterministic inference

## Training Details (from paper)

- Optimizer: Adam, lr=1e-4
- Batch size: 32 (4 GPUs × 8)
- Block shape: 32×32×32
- Data: 11,480 T1 sMRI volumes, 50-class FreeSurfer parcellation
- MC samples at inference: 10
