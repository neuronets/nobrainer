# This Dockerfile was taken largely from TensorFlow's official GPU Dockerfile.
# It has been updated to clear the apt lists after installing to minimize the
# size of the resulting image.

ARG UBUNTU_VERSION=18.04

ARG CUDA=10.0
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.4.1.5-1

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
      nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda${CUDA} \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-command-line-tools-${CUDA/./-} \
        cuda-cublas-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libnvinfer5=5.0.2-1+cuda${CUDA} \
        libzmq3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

ENV LANG="C.UTF-8"
COPY [".", "/opt/nobrainer"]
RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        git \
        python3 \
        python3-h5py \
        python3-numpy \
        python3-pip \
        python3-scipy \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --editable /opt/nobrainer[gpu] \
    && ln -s $(which python3) /usr/local/bin/python \
    && ln -sf /opt/nobrainer/models /models
ENTRYPOINT ["nobrainer"]
LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
