FROM nvcr.io/nvidia/pytorch:24.05-py3

LABEL maintainer="ICAI Comillas"
LABEL project="Algebraic-Reasoning-Distiller"
LABEL description="Multi-agent SAIR distiller pipeline — Qwen2.5-7B LoRA on NVIDIA DGX H200"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    nano \
    tmux \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install project dependencies
# torch / torchvision / torchaudio are pre-installed in the base image (CUDA 12.4)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Working directory — synced with ~/workspace on the DGX host via Docker volume
WORKDIR /root/work

CMD ["/bin/bash"]
