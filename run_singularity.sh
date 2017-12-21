#!/usr/bin/env bash

singularity shell --nv -B /om/user/jakubk:/om/user/jakubk:rw -B /om2:/om2:ro /storage/gablab001/data/singularity-images/nobrainer_2017-12-13.img
