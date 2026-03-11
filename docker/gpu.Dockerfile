FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
COPY [".", "/opt/nobrainer"]
RUN pip install uv \
    && uv pip install --system \
        "/opt/nobrainer[bayesian,generative,versioning]" \
        monai \
        pyro-ppl \
    && rm -rf /root/.cache/uv
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR "/work"
LABEL maintainer="Satrajit Ghosh <satrajit.ghosh@gmail.com>"
LABEL org.opencontainers.image.title="nobrainer-gpu-pytorch"
LABEL org.opencontainers.image.description="nobrainer with PyTorch GPU support"
ENTRYPOINT ["nobrainer"]
