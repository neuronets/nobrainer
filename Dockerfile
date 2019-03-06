FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter
COPY [".", "/opt/nobrainer"]
RUN pip install --no-cache-dir click nibabel numpy pytest scipy scikit-image seaborn \
    && pip install --no-cache-dir --editable /opt/nobrainer
ENTRYPOINT ["jupyter-notebook", "--ip=0.0.0.0", "--no-browser"]
LABEL maintainer="Jakub Kaczmarzyk <jakubk@mit.edu>"
