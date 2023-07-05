#!/bin/bash

singularity exec --nv -B /om ${SINGULARITY_IMAGE} ./run-in-env.sh $@
