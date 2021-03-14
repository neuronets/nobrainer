# Nobrainer in a container

The Dockerfiles in this directory can be used to create Docker images to use _Nobrainer_ on CPU or GPU.

## Build images

```bash
cd /code/nobrainer  # Top-level nobrainer directory
docker build -t neuronets/nobrainer:master-cpu -f docker/cpu.Dockerfile .
docker build -t neuronets/nobrainer:master-gpu -f docker/gpu.Dockerfile .
```

# Convert Docker images to Singularity containers

Using Singularity version 3.x, Docker images can be converted to Singularity containers using the `singularity` command-line tool.

## Pulling from DockerHub

In most cases (e.g., working on a HPC cluster), the _Nobrainer_ singularity container can be created with:

```bash
singularity pull docker://neuronets/nobrainer:master-gpu
```

## Building from local Docker cache

If you built a _Nobrainer_ Docker images locally and would like to convert it to a Singularity container, you can do so with:

```bash
sudo singularity pull docker-daemon://neuronets/nobrainer:master-gpu
```

Please note the use of `sudo` here. This is necessary for interacting with the Docker daemon.
