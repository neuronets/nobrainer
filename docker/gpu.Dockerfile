FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
COPY [".", "/opt/nobrainer"]
RUN pip install --no-cache-dir --editable /opt/nobrainer
WORKDIR "/work"
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
ENTRYPOINT ["nobrainer"]
