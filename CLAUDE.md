# Nobrainer Development Guidelines

## Project Overview

Nobrainer is a PyTorch-based deep learning library for 3D brain MRI segmentation. It provides scikit-learn-style estimators (`Segmentation`, `Generation`), Bayesian models (VWN/FFG, Pyro-based), and a comprehensive data pipeline (MONAI transforms, Zarr3 stores, SynthSeg generation).

## Technology Stack

- **Python**: 3.12+; CI matrix 3.12/3.13/3.14
- **Package management**: `uv` throughout (never pip/conda/poetry)
- **ML framework**: PyTorch >= 2.0
- **Medical imaging**: MONAI >= 1.3 (transforms, losses, metrics, model wrappers)
- **Bayesian**: Pyro-ppl >= 1.9 (optional `[bayesian]` extra)
- **Data**: Zarr >= 3.0 (optional `[zarr]` extra), NIfTI via nibabel
- **Testing**: pytest; pre-commit (black, flake8, isort, codespell)
- **CI**: GitHub Actions; EC2 GPU runner for GPU tests

## Commands

```bash
# Install
uv pip install -e ".[all]"

# Test (CPU)
uv run pytest nobrainer/tests/unit/ -m "not gpu" --tb=short

# SR-tests (somewhat realistic, need sample brain data)
uv run pytest nobrainer/sr-tests/ -m "not gpu"

# Lint
uv run pre-commit run --all-files
```

## Code Conventions

- All models: `(B, C, D, H, W)` input → `(B, n_classes, D, H, W)` output
- Factory functions: `model_name(n_classes=1, in_channels=1, **kwargs) -> nn.Module`
- Bayesian models: `supports_mc = True` class attribute; `forward(x, **kwargs)` accepts `mc=True/False`
- Prediction: use `model_supports_mc(model)` to check, never `try/except TypeError`
- Labels: always squeeze channel dim + cast to `long` before `CrossEntropyLoss`
- Device selection: `nobrainer.gpu.get_device()` (CUDA > MPS > CPU)
- Data augmentation: `TrainableCompose` wraps MONAI Compose; `Augmentation()` wrapper auto-skips during predict

## Key Modules

| Module | Purpose |
|--------|---------|
| `models/` | MeshNet, SegFormer3D, UNet, SwinUNETR, SegResNet, Bayesian variants |
| `processing/` | Segmentation/Generation estimators, Dataset builder |
| `augmentation/` | SynthSeg generator, TrainableCompose, profiles |
| `datasets/` | OpenNeuro fetching, Zarr3 store management |
| `training.py` | `fit()` with DDP, AMP, validation, callbacks |
| `prediction.py` | Block-based predict, strided reassembly, MC uncertainty |
| `losses.py` | Dice, FocalLoss, DiceCE, ELBO, class weights |
| `gpu.py` | Device detection, auto batch size, multi-GPU scaling |
| `slurm.py` | SLURM preemption handler, checkpoint/resume |
| `experiment.py` | Local JSONL/CSV + optional W&B tracking |

## Development Workflow (Speckit Constitution)

When working on new features or significant changes, follow these principles:

### I. Specification-First

Every feature MUST begin with a written specification before implementation:
- Prioritized user stories with independently testable acceptance scenarios
- Functional requirements written as verifiable constraints (MUST/SHOULD)
- Measurable success criteria that are technology-agnostic

### II. Incremental Planning

Plans are built in ordered phases — no phase may be skipped:
- **Phase 0 — Research**: Resolve all unknowns before design
- **Phase 1 — Design**: Data model, interface contracts, quickstart documented
- **Phase 2 — Tasks**: Actionable task list organized by user story priority

Implementation MUST NOT begin until tasks exist.

### III. Independent User-Story Delivery

- Each P1 story MUST produce a viable MVP with standalone value
- Stories MUST NOT have hard runtime dependencies on lower-priority stories
- Tasks MUST be labeled with their owning story (`[US1]`, `[US2]`, etc.)

### IV. Constitution Compliance Gate

Every plan MUST include a Constitution Check evaluated before research and after design. Violations MUST be justified with a simpler alternative explicitly rejected.

### V. Simplicity & YAGNI

- Prefer the simplest architecture that satisfies current user stories
- Do not introduce abstractions for hypothetical future requirements
- Complexity MUST be justified against a concrete, present need

### VI. Git Commit Discipline

- Feature work on dedicated branches (`###-feature-name`)
- Planning artifacts committed after each speckit command
- Each completed task results in at least one commit
- Prefer new commits over amending

### VII. Technology Stack Standards

- Python: `uv` for all environment and package management
- Containers: Docker only
- No substitutions without justified amendment

## Quality Gates

| Gate | Condition |
|------|-----------|
| G1 | spec.md has ≥1 user story with acceptance scenarios |
| G2 | All NEEDS CLARIFICATION resolved before design |
| G3 | Constitution Check passes (or violations justified) |
| G4 | tasks.md exists and all tasks reference a user story |
| G5 | P1 story independently verified before P2 work |
| G6 | All planning artifacts committed to feature branch |

## Speckit Commands (if available)

```
/speckit.specify  → spec.md
/speckit.clarify  → spec.md (revised)
/speckit.plan     → plan.md, research.md, data-model.md, quickstart.md
/speckit.tasks    → tasks.md
/speckit.implement → code
/speckit.analyze  → consistency report
```

If speckit is not installed, follow the principles above manually.
