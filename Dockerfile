FROM tensorflow/tensorflow:1.6.0-gpu-py3

WORKDIR /opt/nobrainer
COPY . .
RUN pip install --no-cache-dir -e /opt/nobrainer[gpu] \
    && mkdir bin \
    && mv vols2hdf5.py train.py train_on_hdf5.py bin/. \
    && chmod +x bin/*.py
    && useradd --no-user-group --create-home --shell /bin/bash neuro \

ENV PATH="$PATH:/opt/nobrainer/bin"

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
