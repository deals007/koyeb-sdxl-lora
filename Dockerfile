FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates unzip \
    python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# (Optional but helpful) accelerate config sometimes expects this
RUN python3 -m pip install --upgrade pip wheel setuptools

# Install PyTorch CUDA 12.1
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Clone kohya trainer
RUN git clone --depth 1 --recurse-submodules https://github.com/bmaltais/kohya_ss.git /workspace/kohya_ss

WORKDIR /workspace/kohya_ss

# Install kohya requirements
RUN python3 -m pip install -r requirements.txt \
 && python3 -m pip install bitsandbytes accelerate transformers safetensors

# Copy dataset zip into the image
COPY dataset/bimbo-nails.zip /workspace/bimbo-nails.zip

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
