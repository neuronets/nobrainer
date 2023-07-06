#!/bin/bash
set -xe
echo "${PWD}"
singularity exec --nv -B /om ${SINGULARITY_IMAGE} ${PWD}/util/run-in-env.sh $@
