#!/usr/bin/env bash

#SBATCH -c10
#SBATCH --mem=8GB
#SBATCH -t0-10:00:00

source activate tumseg

python /home/jakubk/nobrainer/brainbox_downloader/metasearch_to_niftis.py

