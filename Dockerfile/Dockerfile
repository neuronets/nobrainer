# Dockerfile for the brain extraction neural network project.
#
# The vast majority of this file is adapted from Satra's keras-gpu Singularity
# file: https://singularity-hub.org/containers/849/

FROM tensorflow/tensorflow:1.1.0-rc1-gpu-py3

ENV NV_DRIVER_VERSION=375.20 \
    NV_DRIVER_PATH=/usr/local/NVIDIA-Linux-x86_64 \
    LC_ALL=C

WORKDIR /usr/local/
RUN locale-gen en_US.UTF-8 \
    # Install NVIDIA drivers.
    && echo "Installing NVIDIA drivers ..." \
    && NV_URL=http://us.download.nvidia.com/XFree86/Linux-x86_64/${NV_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION}.run \
    && curl -sSL -o /tmp/nvidia_installer_${NV_DRIVER_VERSION}.run $NV_URL \
    && sh /tmp/nvidia_installer_${NV_DRIVER_VERSION}.run -x \
    && mv NVIDIA-Linux-x86_64-${NV_DRIVER_VERSION} NVIDIA-Linux-x86_64 \
    && cd NVIDIA-Linux-x86_64/ \
    # Symlink files with driver version numbers.
    && for n in *.$NV_DRIVER_VERSION; do \
          ln -v -s $n ${n%.${NV_DRIVER_VERSION}}; \
       done \
    && ln -v -s libnvidia-ml.so.${NV_DRIVER_VERSION} libnvidia-ml.so.1 \
    && ln -v -s libcuda.so.${NV_DRIVER_VERSION} libcuda.so.1 \
    # Unset this variable. Otherwise, error is raised:
    # PermissionError: [Errno 13] Permission denied: '/run/user'
    && unset XDG_RUNTIME_DIR \
    # Install Python packages.
    && echo "Installing Python packages ..." \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir h5py keras nibabel \
    # Clean up.
    && rm -rf /tmp/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$NV_DRIVER_PATH:$LD_LIBRARY_PATH \
    PATH=$NV_DRIVER_PATH:$PATH

WORKDIR /home
ENTRYPOINT ["/usr/bin/python"]
