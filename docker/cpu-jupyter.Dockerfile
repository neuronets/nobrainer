FROM kaczmarj/nobrainer:latest
RUN apt-get update \
    && apt-get install --yes --quiet --no-install-recommends \
        graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
        matplotlib \
        notebook \
        pandas \
        pydot \
        seaborn
WORKDIR /notebooks
RUN cp /opt/nobrainer/guide/*.ipynb .
ENTRYPOINT ["jupyter-notebook", "--ip=0.0.0.0", "--no-browser"]
LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
