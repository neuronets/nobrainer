FROM tensorflow/tensorflow:1.4.0-gpu-py3

RUN pip install --no-cache-dir \
        nibabel \
    && useradd --no-user-group --create-home --shell /bin/bash neuro

USER neuro
WORKDIR /home/neuro
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/:$LD_LIBRARY_PATH"
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
