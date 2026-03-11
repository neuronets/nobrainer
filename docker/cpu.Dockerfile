FROM python:3.14-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*
COPY [".", "/opt/nobrainer"]
RUN pip install --no-cache-dir uv \
    && uv pip install --system \
        "torch" \
        "/opt/nobrainer[bayesian,generative]" \
        monai \
        pyro-ppl \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
    && rm -rf /root/.cache/uv
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
WORKDIR "/work"
LABEL maintainer="Satrajit Ghosh <satrajit.ghosh@gmail.com>"
LABEL org.opencontainers.image.title="nobrainer-cpu-pytorch"
LABEL org.opencontainers.image.description="nobrainer with PyTorch CPU-only support"
ENTRYPOINT ["nobrainer"]
