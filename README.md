# neuro_nn

Attempting to create a neural network to identify brains in structural magnetic resonance images.


# Commands

## Get the Docker image as a Singularity image on OpenMind

```shell
$> cd /path/to/your/singularity-images
$> singularity pull docker://kaczmarj/neuro_nn
```

## Run a Jupyter Notebook within the project's Singularity container

```shell
$> singularity shell --nv -B /om/user/jakubk/neuro_nn/data:/data:ro \
-B ~/neuro_nn:/home/neuro_nn /om/user/jakubk/singularity_images/neuro_nn.img
container$> unset XDG_RUNTIME_DIR
container$> jupyter notebook --ip=* --port=9000
```
