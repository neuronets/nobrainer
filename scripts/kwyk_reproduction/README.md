# KWYK Brain Extraction Reproduction

Reproduce the kwyk brain extraction study (McClure et al., Frontiers in
Neuroinformatics 2019) using the refactored PyTorch nobrainer.

**Reference**: https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00067/full

## Current Status

The reproduction pipeline (scripts, infrastructure, CI smoke test) is
**code-complete and CI-verified**. The actual GPU training and comparison
have **not yet been run** — the scripts need to be executed on a machine
with a GPU and sufficient time/data. See "Next Steps" below.

## Quick Setup

```bash
# Option A: Use the orchestrator script (creates venv automatically)
cd scripts/kwyk_reproduction
./run.sh --smoke-test   # Quick verification (5 volumes, 2 epochs)
./run.sh                # Full pipeline

# Option B: Manual setup
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e "../../[bayesian,versioning,dev]" monai pyro-ppl datalad matplotlib pyyaml scipy
```

## Pipeline Steps

### Step 1: Assemble Dataset

```bash
python 01_assemble_dataset.py --datasets ds000114 --output-csv manifest.csv
```

Downloads T1w + aparc+aseg volumes from OpenNeuro fmriprep derivatives via
DataLad. Start with 1 dataset (~10 subjects) for smoke testing, then scale:

```bash
# Scale to more datasets
python 01_assemble_dataset.py \
  --datasets ds000114 ds000228 ds002609 ds001021 ds002105 \
  --output-csv manifest.csv --conform
```

### Step 2: Train Deterministic MeshNet (Warm-Start Foundation)

```bash
python 02_train_meshnet.py --manifest manifest.csv --epochs 50
```

Trains a standard MeshNet with kwyk-matching parameters (filters=96,
block_shape=32³, lr=0.0001). This model's weights serve as the mean
priors for the Bayesian model in Step 3.

**Output**: `checkpoints/meshnet/model.pth`, `figures/meshnet_learning_curve.png`

### Step 3: Train Bayesian MeshNet

```bash
# With warm-start (recommended — faster convergence)
python 03_train_bayesian.py \
  --manifest manifest.csv \
  --warmstart checkpoints/meshnet/model.pth \
  --epochs 50

# Without warm-start (for comparison)
python 03_train_bayesian.py --manifest manifest.csv --no-warmstart --epochs 50
```

Uses ELBO loss (CrossEntropy + KL divergence). The warm-start transfers
deterministic weights to `weight_mu` and initializes `weight_rho` to -3.0
(low initial uncertainty), based on the MOPED method.

**Output**: `checkpoints/bayesian/model.pth`, `checkpoints/bayesian/croissant.json`,
`figures/bayesian_learning_curve.png`

### Step 4: Evaluate

```bash
python 04_evaluate.py \
  --model checkpoints/bayesian/model.pth \
  --manifest manifest.csv --split test --n-samples 10
```

Computes per-volume Dice, saves variance + entropy maps as NIfTI.

### Step 5: Compare with Original KWYK

```bash
python 05_compare_kwyk.py \
  --new-model checkpoints/bayesian/model.pth \
  --kwyk-dir ../../kwyk \
  --manifest manifest.csv
```

Runs the original kwyk container on the same test volumes and generates a
Dice scatter plot + comparison table. **Note**: This requires the kwyk
container at `../../kwyk` to be functional. The comparison is only meaningful
after the Bayesian model has been trained to convergence (Steps 2-3).

### Step 6: Block Size Sweep (Optional)

```bash
python 06_block_size_sweep.py --manifest manifest.csv --block-sizes 32 64 128
```

## Next Steps for GPU Execution

The following steps should be performed on a machine with a GPU (e.g., the
EC2 GPU runner or a local workstation):

### Phase 1: Smoke Test (15 minutes, any GPU)

```bash
./run.sh --smoke-test
```

Verify the pipeline works end-to-end with tiny models. Check
`figures/` for learning curves showing loss decrease.

### Phase 2: Small-Scale Training (1-2 hours, T4 16GB)

```bash
python 01_assemble_dataset.py --datasets ds000114 --output-csv manifest.csv
python 02_train_meshnet.py --manifest manifest.csv --epochs 20
python 03_train_bayesian.py --manifest manifest.csv \
  --warmstart checkpoints/meshnet/model.pth --epochs 20
python 04_evaluate.py --model checkpoints/bayesian/model.pth \
  --manifest manifest.csv --split test --n-samples 10
```

**Expected**: Validation Dice ≥0.80 for brain extraction on 10 subjects.
Check `figures/bayesian_learning_curve.png` for convergence.

### Phase 3: Full Reproduction (8-24 hours, V100 16GB+)

```bash
python 01_assemble_dataset.py \
  --datasets ds000114 ds000228 ds002609 ds001021 ds002105 \
  --output-csv manifest.csv --conform
python 02_train_meshnet.py --manifest manifest.csv --epochs 50
python 03_train_bayesian.py --manifest manifest.csv \
  --warmstart checkpoints/meshnet/model.pth --epochs 50
python 04_evaluate.py --model checkpoints/bayesian/model.pth \
  --manifest manifest.csv --split test --n-samples 10
python 05_compare_kwyk.py --new-model checkpoints/bayesian/model.pth \
  --kwyk-dir ../../kwyk --manifest manifest.csv
```

**Target**: Validation Dice ≥0.90 (kwyk achieved 0.97+ with 11,000 subjects).
Check `figures/dice_scatter.png` for kwyk comparison.

### Phase 4: Scale and Optimize

To approach kwyk's full performance:

1. **Add more datasets**: Add OpenNeuro dataset IDs to the `--datasets` list
2. **Block size sweep**: `python 06_block_size_sweep.py --block-sizes 32 64 128`
3. **SynthSeg augmentation**: `python 03_train_bayesian.py --augmentation mixed`
4. **Longer training**: Increase `--epochs` to 100+

### Phase 5: Automated Hyperparameter Optimization

Use nobrainer's autoresearch loop to explore hyperparameters overnight:

```bash
# Set up the research directory
mkdir -p research/kwyk_bayesian
cp checkpoints/bayesian/model.pth research/kwyk_bayesian/
cat > research/kwyk_bayesian/program.md << 'EOF'
## Exploration Targets
- kl_weight: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
- dropout_rate: 0.0, 0.1, 0.25, 0.5
- filters: 71, 96, 128
- prior_type: standard_normal, laplace
- block_shape: 32, 64
- learning_rate: 1e-5, 5e-5, 1e-4, 5e-4

## Success Criterion
- val_dice improvement over current best
- Max 30 min per experiment
EOF

# Launch overnight optimization
nobrainer research run \
  --working-dir research/kwyk_bayesian \
  --model-family bayesian_meshnet \
  --max-experiments 20 \
  --budget-hours 8
```

The autoresearch loop will:
1. Propose hyperparameter changes (via LLM or random grid)
2. Train, evaluate, keep improvements, revert failures
3. Save the best model with full Croissant-ML provenance

Check results: `cat research/kwyk_bayesian/run_summary.md`

## Configuration

Edit `config.yaml` to change default hyperparameters:

| Parameter | Default | kwyk Original | Notes |
|-----------|---------|---------------|-------|
| filters | 96 | 96 | Feature maps per layer |
| receptive_field | 37 | 37 | Dilation schedule [1,1,1,2,4,8,1] |
| block_shape | [32,32,32] | [32,32,32] | Patch size for training |
| lr | 0.0001 | 0.0001 | Adam learning rate |
| kl_weight | 1.0 | implicit | KL divergence scaling |
| dropout_rate | 0.25 | 0.25 | Spatial dropout |
| n_classes | 2 | 50 | Binary brain extraction (kwyk used 50-class) |
| label_mapping | binary | N/A | Also supports 6/50/115-class |

## Label Mappings

The `label_mappings/` directory contains CSVs that remap FreeSurfer
aparc+aseg codes to target classes:

- **binary**: Any non-zero → 1 (brain extraction)
- **6-class**: Coarse parcellation (WM, cortex, ventricles, cerebellum, etc.)
- **50-class**: Matches original kwyk study
- **115-class**: Fine-grained parcellation

## GPU Requirements

| Task | Block Size | Filters | GPU Memory | Time |
|------|-----------|---------|------------|------|
| Smoke test | 16³ | 16 | ≥4 GB | ~5 min |
| Small training | 32³ | 96 | ≥16 GB | ~2 hr |
| Full reproduction | 32³ | 96 | ≥16 GB | ~24 hr |
| Block sweep 64³ | 64³ | 96 | ≥16 GB | ~4 hr |
| Full-brain 256³ | 256³ | 96 | ≥24 GB | N/A |

## What Has NOT Been Done Yet

- [ ] Actual GPU training (smoke test passes in CI, but full training needs GPU time)
- [ ] Comparison against kwyk container (requires trained model from Steps 2-3)
- [ ] Full-scale reproduction with 100+ subjects
- [ ] Block size sweep results
- [ ] SynthSeg augmentation experiments
- [ ] Autoresearch hyperparameter optimization
