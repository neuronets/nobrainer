FROM tensorflow/tensorflow:1.6.0-gpu-py3

WORKDIR /opt/nobrainer
COPY . .
RUN pip install --no-cache-dir -e /opt/nobrainer \
    && chmod +x vols2hdf5.py train.py train_on_hdf5.py \
    && ln -sv /opt/nobrainer/vols2hdf5.py /usr/local/bin/vols2hdf5.py \
    && ln -sv /opt/nobrainer/train.py /usr/local/bin/train.py \
    && ln -sv /opt/nobrainer/train_on_hdf5.py /usr/local/bin/train_on_hdf5.py \
    && useradd --no-user-group --create-home --shell /bin/bash neuro \
    && ldconfig

USER neuro
WORKDIR /home/neuro
ENTRYPOINT ["/usr/bin/python"]

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
