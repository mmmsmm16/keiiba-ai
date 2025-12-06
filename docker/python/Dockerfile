# RTX 50シリーズ対応のため、ユーザー検証済みのベースイメージを使用 (CUDAライブラリなどが含まれる)
FROM nvcr.io/nvidia/pytorch:24.06-py3

# aptの対話モードを無効化
ENV DEBIAN_FRONTEND=noninteractive

# システム依存パッケージのインストール
# libpq-dev は psycopg2 に必要
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# pipのアップグレード
RUN pip install --no-cache-dir --upgrade pip

# PyTorch Nightly のインストール (RTX 50シリーズ/sm_120 対応の CUDA 12.6+ 版)
# requirements.txt より先にインストールすることで、NumPy 2.x 環境を構築し、
# 後続のライブラリが NumPy 2.x 対応版を選択できるようにする。
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Python依存パッケージのインストール
COPY requirements.txt /tmp/requirements.txt

# PyTorch (torch) は既に Nightly 版を入れているため、requirements.txt からは除外してインストールする。
# これにより、安定版の torch による上書き（および互換性のない CUDA バージョンへの変更）を防ぐ。
RUN grep -v "^torch" /tmp/requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

# pytorch-tabnet のインストール (念のため明示的に)
RUN pip install --no-cache-dir pytorch-tabnet>=4.1.0

# ロケール設定
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# デフォルトコマンド
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
