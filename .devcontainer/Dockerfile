FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.4.18

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/home/user/.local/bin:/home/user/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        ffmpeg \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==${UV_VERSION}

RUN useradd -m -s /bin/bash user
USER user
WORKDIR /home/user/asr

RUN python${PYTHON_VERSION} -m venv /home/user/.venv

COPY --chown=user:user pyproject.toml ./
RUN uv pip install --python /home/user/.venv/bin/python -e ".[dev]" || true

COPY --chown=user:user . .
RUN uv pip install --python /home/user/.venv/bin/python -e ".[dev]"
