FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopenmpi-dev libsndfile1 && \
    rm -rf /var/lib/apt/lists/*
