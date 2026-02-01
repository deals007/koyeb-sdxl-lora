FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates unzip \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip wheel setuptools

# PyTorch CUDA 12.1
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Clone kohya with submodules
RUN git clone --depth 1 --recurse-submodules https://github.com/bmaltais/kohya_ss.git /workspace/kohya_ss

WORKDIR /workspace/kohya_ss
RUN git submodule update --init --recursive

RUN python3 -m pip install -r requirements.txt \
 && python3 -m pip install bitsandbytes accelerate transformers safetensors huggingface_hub

COPY dataset/bimbo-nails.zip /workspace/bimbo-nails.zip
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
