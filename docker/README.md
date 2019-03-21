# Nobrainer in a container

The Dockerfiles in this directory can be used to create Docker images to use _Nobrainer_ on CPU or GPU.

| Dockerfile | Tag |
|---| --- |
| [cpu.Dockerfile](cpu.Dockerfile) | `kaczmarj/nobrainer:latest` |
| [gpu.Dockerfile](gpu.Dockerfile) | `kaczmarj/nobrainer:latest-gpu` |
| [cpu-jupyter.Dockerfile](cpu-jupyter.Dockerfile) | `kaczmarj/nobrainer:latest-jupyter` |
| [gpu-jupyter.Dockerfile](gpu-jupyter.Dockerfile) | `kaczmarj/nobrainer:latest-gpu-jupyter` |


## Build images

```bash
cd /code/nobrainer  # Top-level nobrainer directory
# Build containers with bare necessities.
docker build -t kaczmarj/nobrainer:latest -f docker/cpu.Dockerfile .
docker build -t kaczmarj/nobrainer:latest-gpu -f docker/gpu.Dockerfile .
# Build containers with jupyter.
docker build -t kaczmarj/nobrainer:latest-jupyter -f docker/cpu-jupyter.Dockerfile .
docker build -t kaczmarj/nobrainer:latest-gpu-jupyter -f docker/gpu-jupyter.Dockerfile .
```


# Convert Docker images to Singularity containers

Using Singularity version 3.x, Docker images can be converted to Singularity containers using the `singularity` command-line tool.

## Pulling from DockerHub

In most cases (e.g., working on a HPC cluster), the Nobrainer singularity container can be created with:

```bash
singularity pull docker://kaczmarj/nobrainer:latest-gpu-jupyter
```

## Building from local Docker cache

If you built a Nobrainer Docker images locally and would like to convert it to a Singularity container, you can do so with:

```bash
sudo singularity pull docker-daemon://kaczmarj/nobrainer:latest-gpu-jupyter
```

Please note the use of `sudo` here. This is necessary for interacting with the Docker daemon.
