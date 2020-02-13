FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
ARG DEBIAN_FRONTEND="noninteractive"
RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
        matplotlib \
        pandas \
        pydot \
        seaborn
COPY [".", "/opt/nobrainer"]
RUN pip install --no-cache-dir --editable /opt/nobrainer
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
ENTRYPOINT ["nobrainer"]
