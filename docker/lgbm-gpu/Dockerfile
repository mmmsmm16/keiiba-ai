# RTX 50シリーズ対応のため、ベースイメージは app コンテナと同じものを使用
FROM nvcr.io/nvidia/pytorch:24.06-py3

# aptの対話モードを無効化
ENV DEBIAN_FRONTEND=noninteractive
# RTX 50 シリーズの未サポート警告によるエラーを回避
ENV NVIDIA_DISABLE_REQUIRE=true
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Build dependencies for LightGBM GPU
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    git \
    cmake \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-test-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# pipのアップグレード
RUN pip install --no-cache-dir --upgrade pip

# PyTorch 2.7.0 安定版のインストール (Blackwell/RTX 50シリーズ/sm_120 公式対応)
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Python依存パッケージのインストール (既存の app と同じものを使用)
COPY requirements.txt /tmp/requirements.txt
RUN grep -v "^torch" /tmp/requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# LightGBM を CUDA 対応でビルド・インストール
RUN pip install --no-cache-dir cmake ninja scikit-build-core
RUN pip uninstall -y lightgbm && \
    export MAKEFLAGS="-j$(nproc --all | awk '{print ($1>2?2:$1)}')" && \
    CMAKE_ARGS="-DUSE_CUDA=1 -G Ninja" pip install lightgbm==4.6.0 --no-cache-dir --no-binary lightgbm

# pytorch-tabnet のインストール
RUN pip install --no-cache-dir pytorch-tabnet>=4.1.0

# ロケール設定
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# SHMEM制限のための設定 (NVIDIA推奨)
# note: docker-compose 側で ipc: host を設定することが望ましい

# デフォルトコマンド (Jupyter Lab を起動)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
