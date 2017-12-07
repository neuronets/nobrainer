# Dockerfile for 3D semantic segmentation project.
#
# - Use `singularity pull ...` to convert the resulting docker image to a
#   singularity image.
# - Use singularity >= 2.3.0 to mount local NVIDIA drivers with the --nv option.

FROM tensorflow/tensorflow:1.4.0-gpu-py3

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir h5py \
                                  keras \
                                  nibabel \
                                  numpy

WORKDIR /home
ENTRYPOINT ["/usr/bin/python"]
