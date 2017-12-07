# Dockerfile for the brain extraction neural network project.
#
# - Use `singularity pull ...` to convert the resulting docker image to a
#   singularity image.
# - Use singularity >= 2.3.0 to mount local NVIDIA drivers with the --nv option.

FROM tensorflow/tensorflow:1.2.0-gpu-py3

LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"

# This is not set automatically for some reason. Causes `import tensorflow` to break.
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir h5py \
                                  keras \
                                  nibabel \
                                  opencv-python

# Install NiftyNet
RUN pip install --no-cache-dir packaging \
    && pip install --no-cache-dir niftynet

# Install DLTK
RUN mkdir /opt/dltk \
    && curl -sSL https://github.com/DLTK/DLTK/tarball/master \
    | tar xz -C /opt/dltk --strip-components=1 \
    && pip install -e /opt/dltk \
    && chmod 777 -R /opt/dltk

RUN curl -sSL https://github.com/ellisdg/3DUnetCNN/tarball/master \
    | tar xz -C /opt \
    && curl -sSL https://github.com/ncullen93/Unet-ants/tarball/master \
    | tar xz -C /opt \
    && chmod 777 -R /opt

WORKDIR /home
ENTRYPOINT ["/usr/bin/python"]
