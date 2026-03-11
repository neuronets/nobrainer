FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
COPY [".", "/opt/nobrainer"]
RUN pip install uv \
    && uv pip install --system \
        "/opt/nobrainer[bayesian,generative]" \
        monai \
        pyro-ppl \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/uv
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR "/work"
LABEL maintainer="Satrajit Ghosh <satrajit.ghosh@gmail.com>"
LABEL org.opencontainers.image.title="nobrainer-cpu-pytorch"
LABEL org.opencontainers.image.description="nobrainer with PyTorch CPU support"
ENTRYPOINT ["nobrainer"]
