#!/bin/bash
# SynthSeg Evaluation Pipeline Orchestrator
#
# Usage:
#   ./run.sh --smoke-test         # Quick test (2 epochs, 1 model)
#   ./run.sh                      # Full evaluation
#   ./run.sh --config custom.yaml # Custom config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-config.yaml}"
SMOKE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test) SMOKE=true; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) shift ;;
    esac
done

cd "$SCRIPT_DIR"

echo "=== SynthSeg Evaluation Pipeline ==="
echo "Config: $CONFIG"
echo "Smoke test: $SMOKE"

# Use sample data if no manifest exists
if [ ! -f manifest.csv ]; then
    echo "=== Creating manifest from sample data ==="
    python -c "
import csv
from nobrainer.utils import get_data
src = get_data()
pairs = []
with open(src) as f:
    r = csv.reader(f); next(r)
    pairs = list(r)[:5]
splits = ['train','train','train','val','test']
with open('manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, ['t1w_path','label_path','split']); w.writeheader()
    for i,(t1,lbl) in enumerate(pairs):
        w.writerow(dict(t1w_path=t1, label_path=lbl, split=splits[i]))
print('Manifest created with', len(pairs), 'volumes')
"
fi

if [ "$SMOKE" = true ]; then
    echo "=== Smoke test: 2 epochs, attention_unet, real+mixed ==="
    for mode in real mixed; do
        echo "  Training attention_unet ($mode)..."
        python 02_train.py --config "$CONFIG" --mode "$mode" --model attention_unet \
            --epochs 2 --manifest manifest.csv
    done
    for mode in real mixed; do
        echo "  Evaluating attention_unet ($mode)..."
        python 03_evaluate.py --model "checkpoints/attention_unet_${mode}" \
            --manifest manifest.csv --config "$CONFIG" || true
    done
    python 04_compare.py --results-dir checkpoints/ --output-dir results/ || true
else
    # Full evaluation from config
    MODELS=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(c['training']['modes']))")
    MODES=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(c['models']))")

    for model in $MODES; do
        for mode in $MODELS; do
            echo "=== Training $model ($mode) ==="
            python 02_train.py --config "$CONFIG" --mode "$mode" --model "$model" \
                --manifest manifest.csv
        done
    done

    for model in $MODES; do
        for mode in $MODELS; do
            echo "=== Evaluating $model ($mode) ==="
            python 03_evaluate.py --model "checkpoints/${model}_${mode}" \
                --manifest manifest.csv --config "$CONFIG" || true
        done
    done

    python 04_compare.py --results-dir checkpoints/ --output-dir results/
fi

echo "=== Done. Results in results/ ==="
