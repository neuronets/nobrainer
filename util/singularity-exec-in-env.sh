#!/bin/bash
set -xe
echo "${PWD}"
singularity exec --nv -B /om ${SINGULARITY_IMAGE} ./run-in-env.sh $@
