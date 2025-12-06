# Use user-verified base image for RTX 50 series support (provides CUDA libs etc.)
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# libpq-dev is required for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install requirements, EXCLUDING torch.
# We will install the specific Nightly version of torch later.
RUN grep -v "^torch" /tmp/requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# Explicitly ensure pytorch-tabnet is installed
RUN pip install --no-cache-dir pytorch-tabnet>=4.1.0

# FORCE install PyTorch Nightly with CUDA 12.6 support for RTX 50 series (sm_120)
# This overwrites the PyTorch version that came with the base image or was pulled by dependencies.
# We use --force-reinstall to be absolutely sure.
RUN pip install --no-cache-dir --pre --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# FORCE downgrade NumPy to <2.0 to ensure compatibility with Pandas/PyArrow
# This is necessary because the PyTorch Nightly install above pulls in NumPy 2.x,
# which breaks other libraries compiled for NumPy 1.x.
RUN pip install --no-cache-dir "numpy<2.0"

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
