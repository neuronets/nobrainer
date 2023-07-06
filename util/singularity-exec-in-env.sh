#!/bin/bash
set -xe
singularity exec --nv -B /om ${SINGULARITY_IMAGE} ${WORKDIR}/util/run-in-env.sh $@
