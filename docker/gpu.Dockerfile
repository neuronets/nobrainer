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
# CUDNN version should be updated regularly. Check the TensorFlow GPU
# Dockerfiles for correct version number.
ARG CUDNN=7.6.2.24-1

ARG DEBIAN_FRONTEND="noninteractive"

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

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

ENV LANG="C.UTF-8"
COPY [".", "/opt/nobrainer"]
RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 - \
    && apt-get autoremove --yes --quiet --purge curl \
    && pip3 install --no-cache-dir --editable /opt/nobrainer[gpu] \
    && ln -s $(which python3) /usr/local/bin/python \
    && ln -sf /opt/nobrainer/models /models
ENTRYPOINT ["nobrainer"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
