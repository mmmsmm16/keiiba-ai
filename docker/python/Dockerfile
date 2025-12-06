# Use newer NVIDIA CUDA base image for RTX 50 series support
# CUDA 12.4+ is likely required for sm_120
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch Nightly with CUDA 12.6 support (or latest stable compatible with new GPUs)
# RTX 50 series (sm_120) needs very recent CUDA support.
# We use the nightly build to ensure compatibility.
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Install other requirements
# Exclude torch from requirements.txt to avoid overwriting
RUN grep -v "torch" /tmp/requirements.txt > /tmp/requirements_no_torch.txt
RUN pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
