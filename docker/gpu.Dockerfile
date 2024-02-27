FROM tensorflow/tensorflow:2.15.0.post1-gpu-jupyter
COPY [".", "/opt/nobrainer"]
RUN cd /opt/nobrainer
RUN python3 -m pip install --no-cache-dir /opt/nobrainer
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR "/work"
LABEL maintainer="Satrajit Ghosh <satrajit.ghosh@gmail.com>"
ENTRYPOINT ["nobrainer"]
