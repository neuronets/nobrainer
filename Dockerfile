FROM tensorflow/tensorflow:1.6.0-gpu-py3

WORKDIR /opt/nobrainer
COPY . .
RUN pip install --no-cache-dir -e /opt/nobrainer \
    && chmod +x vols2hdf5.py train.py train_on_hdf5.py \
    && ln -sv vols2hdf5.py /usr/bin/vols2hdf5.py \
    && ln -sv train.py /usr/bin/train.py \
    && ln -sv train_on_hdf5.py /usr/bin/train_on_hdf5.py \
    && useradd --no-user-group --create-home --shell /bin/bash neuro

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
