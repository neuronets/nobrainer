#!/bin/bash
set -xe
echo "${PWD}"
singularity exec --nv -B /om ${SINGULARITY_IMAGE} util/run-in-env.sh $@
