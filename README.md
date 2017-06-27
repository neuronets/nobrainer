# neuro_nn

Attempting to create a neural network to identify brains in structural magnetic resonance images.


# Procedure

1. Using anatomical volumes and 3D brain masks from http://openneu.ro/metasearch/, construct an HDF5 file with anatomical and brain mask slices. Start with axial slices, but add more in the future.


# Commands

## Get the Docker image as a Singularity image on OpenMind

```shell
$> cd /path/to/your/singularity-images
$> singularity pull docker://kaczmarj/neuro_nn
```

## Run a Jupyter Notebook within the project's Singularity container

```shell
om$> singularity shell -B /om/user/jakubk/neuro_nn/data:/data:ro \
-B /home/jakubk/neuro_nn:/home/neuro_nn \
/path/to/your/singularity-images/image.img \
container$> jupyter notebook --ip=* --port=9000
```
