FROM debian:buster-slim
ARG DEBIAN_FRONTEND="noninteractive"
ENV LANG="C.UTF-8"
COPY [".", "/opt/nobrainer"]
RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 - \
    && apt-get autoremove --yes --quiet --purge ca-certificates curl \
    && pip3 install --no-cache-dir --editable /opt/nobrainer[cpu] \
    && ln -s $(which python3) /usr/local/bin/python \
    && ln -sf /opt/nobrainer/models /models
ENTRYPOINT ["nobrainer"]
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@gmail.com>"
