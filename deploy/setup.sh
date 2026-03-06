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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# 1. Системные пакеты
# -----------------------------------------------------------------------------
if ! command -v python3.12 &>/dev/null || ! python3.12 -c "import ensurepip" &>/dev/null 2>&1; then
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
else
  log "Python 3.12 already installed, skipping apt"
fi

# -----------------------------------------------------------------------------
# 2. Python окружение
# -----------------------------------------------------------------------------
cd "$REPO_DIR"

FLWR_OK=false
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  installed=$(python -c "import flwr; print(flwr.__version__)" 2>/dev/null || echo "")
  [[ "$installed" == "1.22.0" ]] && FLWR_OK=true
fi

if [[ "$FLWR_OK" == false ]]; then
  log "Creating/updating Python venv..."
  python3.12 -m venv .venv
  source .venv/bin/activate

  pip install --upgrade pip setuptools wheel -q

  log "Installing PyTorch (CPU)..."
  pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu -q

  log "Installing project dependencies..."
  pip install \
    "flwr[simulation]==1.22.0" \
    "flwr-datasets[vision]>=0.5.0" \
    torchmetrics==1.7.1 \
    numpy pandas scikit-learn matplotlib pyyaml pillow==11.2.1 tqdm rich \
    pytest ruff -q
else
  log "venv with flwr==1.22.0 already present, skipping pip install"
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
