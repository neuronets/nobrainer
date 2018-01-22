#!/usr/bin/env bash

#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --mem=16GB
#SBATCH -t4-00:00:00
#SBATCH -c1

singularity exec --nv -B /om/user/jakubk:/om/user/jakubk:rw -B /om2:/om2:ro /storage/gablab001/data/singularity-images/nobrainer_2018-01-08.sqfs python /om/user/jakubk/nobrainer-code/niftynet_to_keras/train.py

