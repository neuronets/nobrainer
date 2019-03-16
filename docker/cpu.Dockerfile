FROM debian:buster-slim
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
    && pip3 install --no-cache-dir --editable /opt/nobrainer[cpu] \
    && ln -s $(which python3) /usr/local/bin/python
ENTRYPOINT ["nobrainer"]
LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
