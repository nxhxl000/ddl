#!/usr/bin/env bash
# =============================================================================
# deploy/setup.sh — установка окружения на одну VM (сервер или клиент)
# Запускать: bash ~/ddl/deploy/setup.sh [--skip-data]
#
# --skip-data   не скачивать датасеты (полезно на сервере, если данные
#               нужны только клиентам, или для ускорения повторной установки)
# =============================================================================
set -euo pipefail

REPO_DIR="$HOME/ddl"
SKIP_DATA=false

for arg in "$@"; do
  [[ "$arg" == "--skip-data" ]] && SKIP_DATA=true
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# 1. Системные пакеты
# -----------------------------------------------------------------------------
log "Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y -q \
  software-properties-common \
  tmux \
  htop \
  git \
  curl \
  ca-certificates

log "Adding deadsnakes PPA for Python 3.12..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -q
sudo apt-get install -y -q \
  python3.12 \
  python3.12-venv \
  python3.12-dev

# -----------------------------------------------------------------------------
# 2. Python окружение
# -----------------------------------------------------------------------------
log "Creating Python venv..."
cd "$REPO_DIR"
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel -q

# Устанавливаем CPU-версию torch явно (без CUDA), чтобы не качать лишние 2+ GB
log "Installing PyTorch (CPU)..."
pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cpu -q

# Остальные зависимости (torch уже стоит, повторная установка пропустит его)
log "Installing project dependencies..."
pip install \
  "flwr[simulation]==1.22.0" \
  "flwr-datasets[vision]>=0.5.0" \
  torchmetrics==1.7.1 \
  numpy pandas scikit-learn matplotlib pyyaml pillow==11.2.1 tqdm rich \
  pytest ruff -q

# -----------------------------------------------------------------------------
# 3. Датасеты
# -----------------------------------------------------------------------------
if [[ "$SKIP_DATA" == false ]]; then
  log "Downloading datasets (MNIST + CIFAR-10)..."
  python -m scripts.download_datasets
else
  log "Skipping dataset download (--skip-data)"
fi

# -----------------------------------------------------------------------------
# 4. Итог
# -----------------------------------------------------------------------------
log "================================================="
log "Setup complete on $(hostname)"
log "Python: $(python --version)"
log "flwr:   $(python -c 'import flwr; print(flwr.__version__)')"
log "torch:  $(python -c 'import torch; print(torch.__version__)')"
log "================================================="
