#!/bin/bash
# KWYK Brain Extraction Reproduction — Full Pipeline Runner
#
# Usage:
#   ./run.sh                    # Full pipeline (data + train + evaluate)
#   ./run.sh --smoke-test       # Quick smoke test (5 volumes, 2 epochs)
#   ./run.sh --step data        # Run only data assembly
#   ./run.sh --step train       # Run only training (deterministic + Bayesian)
#   ./run.sh --step evaluate    # Run only evaluation
#   ./run.sh --step compare     # Run only kwyk comparison
#   ./run.sh --step sweep       # Run only block size sweep
#
# Environment:
#   Creates a dedicated venv at .venv-kwyk/ with all dependencies.
#   Set NOBRAINER_ROOT to override the nobrainer repo location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOBRAINER_ROOT="${NOBRAINER_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="$SCRIPT_DIR/.venv-kwyk"
STEP="${1:---all}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[kwyk]${NC} $*"; }
warn() { echo -e "${YELLOW}[kwyk]${NC} $*"; }
err() { echo -e "${RED}[kwyk]${NC} $*" >&2; }

# --- Setup venv ---
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment at $VENV_DIR"
        uv venv --python 3.14 "$VENV_DIR"
    fi

    log "Installing dependencies..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    uv pip install -e "$NOBRAINER_ROOT[bayesian,zarr,versioning,dev]" \
        monai pyro-ppl datalad matplotlib pyyaml scipy nibabel 2>&1 | tail -3
    log "Dependencies installed"
}

# --- Parse arguments ---
SMOKE_TEST=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --all)
            STEP="--all"
            shift
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

setup_venv
cd "$SCRIPT_DIR"

# --- Smoke test configuration ---
if [ "$SMOKE_TEST" = true ]; then
    log "Running SMOKE TEST (5 volumes, 2 epochs, tiny model)"
    EXTRA_ARGS="--epochs 2"
    DATASETS="ds000114"
    # Use get_data() instead of DataLad for smoke test
else
    EXTRA_ARGS=""
    DATASETS="ds000114 ds000228 ds002609"
fi

# --- Step: Data Assembly ---
run_data() {
    log "Step 1: Assembling dataset from OpenNeuro..."
    python 01_assemble_dataset.py \
        --datasets $DATASETS \
        --output-csv manifest.csv \
        --output-dir data \
        --label-mapping binary
    log "Dataset assembled: $(wc -l < manifest.csv) subjects"
}

# --- Step: Training ---
run_train() {
    log "Step 2: Training deterministic MeshNet (warm-start foundation)..."
    python 02_train_meshnet.py \
        --manifest manifest.csv \
        --config config.yaml \
        --output-dir checkpoints/meshnet \
        $EXTRA_ARGS
    log "Deterministic MeshNet trained (bwn / MAP variant)"

    log "Step 3a: MC Bernoulli dropout variant (bwn_multi)..."
    python 03_train_bayesian.py \
        --manifest manifest.csv \
        --config config.yaml \
        --variant bwn_multi \
        --warmstart checkpoints/meshnet \
        --output-dir checkpoints/bwn_multi \
        $EXTRA_ARGS
    log "MC Bernoulli dropout variant saved"

    log "Step 3b: Spike-and-slab dropout variant (bvwn_multi_prior)..."
    python 03_train_bayesian.py \
        --manifest manifest.csv \
        --config config.yaml \
        --variant bvwn_multi_prior \
        --warmstart checkpoints/meshnet \
        --output-dir checkpoints/bvwn_multi_prior \
        $EXTRA_ARGS
    log "Spike-and-slab dropout variant trained"

    log "Step 3c: Standard Gaussian Bayesian variant (for comparison)..."
    python 03_train_bayesian.py \
        --manifest manifest.csv \
        --config config.yaml \
        --variant bayesian_gaussian \
        --warmstart checkpoints/meshnet \
        --output-dir checkpoints/bayesian_gaussian \
        $EXTRA_ARGS
    log "Gaussian Bayesian variant trained"
}

# --- Step: Evaluate ---
run_evaluate() {
    log "Step 4: Evaluating all model variants on test set..."
    if [ -f 04_evaluate.py ]; then
        for variant_dir in checkpoints/meshnet checkpoints/bwn_multi checkpoints/bvwn_multi_prior checkpoints/bayesian_gaussian; do
            variant_name=$(basename "$variant_dir")
            if [ -f "$variant_dir/model.pth" ]; then
                log "  Evaluating $variant_name..."
                python 04_evaluate.py \
                    --model "$variant_dir/model.pth" \
                    --manifest manifest.csv \
                    --split test \
                    --n-samples 10 \
                    --output-dir "results/$variant_name"
            else
                warn "  Skipping $variant_name (no model.pth found)"
            fi
        done
    else
        warn "04_evaluate.py not found"
    fi
}

# --- Step: Compare ---
run_compare() {
    log "Step 5: Comparing with original kwyk container..."
    if [ -f scripts/kwyk_reproduction/05_compare_kwyk.py ]; then
        python 05_compare_kwyk.py \
            --new-model checkpoints/bayesian/model.pth \
            --kwyk-dir "$NOBRAINER_ROOT/../kwyk" \
            --manifest manifest.csv \
            --output-dir results/comparison
    else
        warn "05_compare_kwyk.py not yet implemented"
    fi
}

# --- Step: Block Size Sweep ---
run_sweep() {
    log "Step 6: Block size sweep..."
    if [ -f scripts/kwyk_reproduction/06_block_size_sweep.py ]; then
        python 06_block_size_sweep.py \
            --manifest manifest.csv \
            --block-sizes 32 64 128 \
            --output-dir results/sweep
    else
        warn "06_block_size_sweep.py not yet implemented"
    fi
}

# --- Execute ---
case "$STEP" in
    --all)
        run_data
        run_train
        run_evaluate
        run_compare
        run_sweep
        ;;
    data)
        run_data
        ;;
    train)
        run_train
        ;;
    evaluate)
        run_evaluate
        ;;
    compare)
        run_compare
        ;;
    sweep)
        run_sweep
        ;;
    *)
        err "Unknown step: $STEP"
        err "Available: data, train, evaluate, compare, sweep"
        exit 1
        ;;
esac

log "Done! Check figures/ and results/ for outputs."
