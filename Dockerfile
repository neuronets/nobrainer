# Nobrainer container specification.

ARG TF_VERSION="1.12.0"
# Use "gpu-py3" to build GPU-enabled container and "py3" for non-GPU container.
ARG TF_ENV="gpu-py3"
FROM tensorflow/tensorflow:${TF_VERSION}-${TF_ENV}

COPY . /opt/nobrainer
RUN \
    # Extras do not have to be installed because the only extra is tensorflow,
    # which is installed in the base image.
    pip install --no-cache-dir -e /opt/nobrainer \
    && rm -rf ~/.cache/pip/* \
    && useradd --no-user-group --create-home --shell /bin/bash neuro

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
