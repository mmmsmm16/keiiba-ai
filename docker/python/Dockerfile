# Use user-verified base image for RTX 50 series support
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

# Install requirements, EXCLUDING torch to avoid overwriting the base image's version.
# The base image (nvcr.io/nvidia/pytorch:24.06-py3) comes with a pre-installed, optimized PyTorch.
# We must ensure we don't accidentally downgrade it with a standard pip install.
RUN grep -v "^torch" /tmp/requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# Explicitly ensure pytorch-tabnet is installed.
# We install it separately to ensure it doesn't trigger a torch reinstall if possible,
# though pip usually respects installed versions.
RUN pip install --no-cache-dir pytorch-tabnet>=4.1.0

# Set locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
