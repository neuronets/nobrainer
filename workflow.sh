#!/bin/bash
#SBATCH -t 100
#SBATCH --mem=8G
#SBATCH --gres=gpu:2

set -ex

TRUTH=${HOME}/data/example
export DATADIR=$(mktemp -d)

wget -P ${DATADIR} -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz

sing-exec ./run.sh \
          nobrainer predict --verbose \
          --model=${HOME}/projects/trained-models/neuronets/brainy/0.1.0/weights/brain-extraction-unet-128iso-model.h5 \
          ${DATADIR}/T1w.nii.gz \
          ${DATADIR}/brainmask.nii.gz
# cmp ${DATADIR}/brainmask.nii.gz ${TRUTH}/brainmask.nii.gz

sing-exec ./run.sh \
          nobrainer predict --verbose \
          --largest-label \
          --model=${HOME}/projects/trained-models/neuronets/brainy/0.1.0/weights/brain-extraction-unet-128iso-model.h5 \
          ${DATADIR}/T1w.nii.gz \
          ${DATADIR}/brainmask-largest.nii.gz
# cmp ${DATADIR}/brainmask-largest.nii.gz ${TRUTH}/brainmask-largest.nii.gz

sing-exec ./run.sh \
          nobrainer predict --verbose \
          --rotate-and-predict \
          --model=${HOME}/projects/trained-models/neuronets/brainy/0.1.0/weights/brain-extraction-unet-128iso-model.h5 \
          ${DATADIR}/T1w.nii.gz \
          ${DATADIR}/brainmask-rotate.nii.gz
# cmp ${DATADIR}/brainmask-rotate.nii.gz ${TRUTH}/brainmask-rotate.nii.gz

sing-exec ./run.sh \
          nobrainer predict --verbose \
          --largest-label \
          --rotate-and-predict \
          --model=${HOME}/projects/trained-models/neuronets/brainy/0.1.0/weights/brain-extraction-unet-128iso-model.h5 \
          ${DATADIR}/T1w.nii.gz \
          ${DATADIR}/brainmask-largest-rotate.nii.gz
# cmp ${DATADIR}/brainmask-largest-rotate.nii.gz ${TRUTH}/brainmask-largest-rotate.nii.gz

sing-exec ./run.sh \
          nobrainer generate --verbose \
          --model=${HOME}/projects/trained-models/neuronets/braingen/0.1.0 \
          --output-shape=128 128 128 ${DATADIR}/generated.nii.gz


# TODO what is the proper evaluation scheme when generation is non-deterministic?
# cmp ${DATADIR}/generated.nii.gz ${TRUTH}/generated.nii.gz

for nb in $(ls ${HOME}/projects/nobrainer/guide/[0-9][0-9]-*.ipynb); do
    sing-exec ./run.sh \
              jupyter nbconvert \
              --execute ${nb} \
              --to html \
              --output-dir ${DATADIR}
    #cat ${DATADIR}/01-getting_started.html
done

rm -rf ${DATADIR}
