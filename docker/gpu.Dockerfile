FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
COPY [".", "/opt/nobrainer"]
RUN python3 -m pip install --no-cache-dir --editable /opt/nobrainer
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR "/work"
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
ENTRYPOINT ["nobrainer"]
