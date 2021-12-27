FROM tensorflow/tensorflow:2.7.0-gpu-jupyter
RUN curl -sSL http://neuro.debian.net/lists/bionic.us-nh.full | tee /etc/apt/sources.list.d/neurodebian.sources.list \
  && export GNUPGHOME="$(mktemp -d)" \
  && echo "disable-ipv6" >> ${GNUPGHOME}/dirmngr.conf \
  && (apt-key adv --homedir $GNUPGHOME --recv-keys --keyserver hkp://pgpkeys.eu 0xA5D32F012649A5A9 \
  || { curl -sSL http://neuro.debian.net/_static/neuro.debian.net.asc | apt-key add -; } ) \
  && apt-get update \
  && apt-get install -y git-annex-standalone git \
  && rm -rf /tmp/*
COPY [".", "/opt/nobrainer"]
RUN python3 -m pip install --no-cache-dir /opt/nobrainer datalad datalad-osf
RUN git config --global user.email "neuronets@example.com" \
    && git config --global user.name "Neuronets maintainers"
RUN datalad clone https://github.com/neuronets/trained-models /models \
  && cd /models && git-annex enableremote osf-storage \
  && datalad get -r .
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR "/work"
LABEL maintainer="Satrajit Ghosh <satrajit.ghosh@gmail.com>"
ENTRYPOINT ["nobrainer"]
