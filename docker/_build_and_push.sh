#!/usr/bin/env bash
#
# Build Nobrainer Docker images and push to DockerHub.

set -ex

if [ ! -d "nobrainer" ]; then
  echo "error: this script must be run in the top-level nobrainer directory"
  exit 1
fi

# Build containers with bare necessities.
docker build -t kaczmarj/nobrainer:latest -f docker/cpu.Dockerfile .
docker build -t kaczmarj/nobrainer:latest-gpu -f docker/gpu.Dockerfile .

# Build containers with jupyter.
docker build -t kaczmarj/nobrainer:latest-jupyter -f docker/cpu-jupyter.Dockerfile .
docker build -t kaczmarj/nobrainer:latest-gpu-jupyter -f docker/gpu-jupyter.Dockerfile .

# Push images
docker push kaczmarj/nobrainer:latest
docker push kaczmarj/nobrainer:latest-gpu
docker push kaczmarj/nobrainer:latest-jupyter
docker push kaczmarj/nobrainer:latest-gpu-jupyter
