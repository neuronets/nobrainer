#!/bin/bash
#SBATCH --job-name=kwyk-zarr
#SBATCH --partition=mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-zarr-%j.out
#SBATCH --error=slurm-zarr-%j.err
#
# Convert PAC NIfTI dataset to Zarr3 for fast chunk-aligned I/O
#
set -euo pipefail

WORK_DIR="/orcd/scratch/orcd/013/satra/kwyk_reproduction"
VENV_DIR="/orcd/data/satra/002/projects/nobrainer/venvs/nobrainer"
MANIFEST="kwyk_manifest_500.csv"
ZARR_OUT="$WORK_DIR/data/kwyk_500.zarr"

cd "$WORK_DIR"
source "${VENV_DIR}/bin/activate"

echo "=== Converting PAC dataset to Zarr3 ==="
echo "Started: $(date)"

# Build image/label path lists from manifest
python -c "
import csv
images, labels = [], []
with open('${MANIFEST}') as f:
    for row in csv.DictReader(f):
        if row['split'] in ('train', 'val', 'test'):
            images.append(row['t1w_path'])
            labels.append(row['label_path'])

# Write temp files for the CLI
with open('/tmp/zarr_images.txt', 'w') as f:
    f.write('\n'.join(images))
with open('/tmp/zarr_labels.txt', 'w') as f:
    f.write('\n'.join(labels))
print(f'Volumes: {len(images)}')
"

# Convert using nobrainer API directly
python -c "
from nobrainer.datasets.zarr_store import create_zarr_store
import csv

pairs = []
subject_ids = []
with open('${MANIFEST}') as f:
    for row in csv.DictReader(f):
        pairs.append((row['t1w_path'], row['label_path']))
        subject_ids.append(row['subject_id'])

print(f'Converting {len(pairs)} volumes to Zarr3...')
store = create_zarr_store(
    image_label_pairs=pairs,
    output_path='${ZARR_OUT}',
    subject_ids=subject_ids,
    chunk_shape=(32, 32, 32),
    conform=True,
    target_shape=(256, 256, 256),
    target_voxel_size=(1.0, 1.0, 1.0),
)
print(f'Zarr store created: {store}')

# Create partition JSON for train/val/test splits
import json
partitions = {'train': [], 'val': [], 'test': []}
with open('${MANIFEST}') as f:
    for row in csv.DictReader(f):
        partitions[row['split']].append(row['subject_id'])

part_path = '${ZARR_OUT}_partition.json'
with open(part_path, 'w') as f:
    json.dump({'partitions': partitions}, f, indent=2)
print(f'Partition file: {part_path}')
for k, v in partitions.items():
    print(f'  {k}: {len(v)} subjects')
"

echo "=== Done: $(date) ==="
