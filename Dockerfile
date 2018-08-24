# Nobrainer container specification.

# Use "gpu-py3" to build GPU-enabled container and "py3" for non-GPU container.
ARG TF_ENV="gpu-py3"
FROM tensorflow/tensorflow:1.9.0-${TF_ENV}

WORKDIR /opt/nobrainer
COPY . .
RUN apt-get update -qq \
    && apt-get install -yq --no-install-recommends \
          imagemagick \
    && apt-get clean \
    # Extras do not have to be installed because the only extra is tensorflow,
    # which is installed in the base image.
    && pip install --no-cache-dir -e /opt/nobrainer \
    && pip install --no-cache-dir \
        https://www.github.com/keras-team/keras-contrib/tarball/master \
        scikit-image \
        seaborn \
    && rm -rf ~/.cache/pip/* \
    && useradd --no-user-group --create-home --shell /bin/bash neuro

ENV PATH="$PATH:/opt/nobrainer/bin"

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
