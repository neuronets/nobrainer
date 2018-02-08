FROM tensorflow/tensorflow:1.6.0-rc0-gpu-py3

RUN pip install --no-cache-dir \
        nibabel \
    && useradd --no-user-group --create-home --shell /bin/bash neuro

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
