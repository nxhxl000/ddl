#!/usr/bin/env bash
# =============================================================================
# deploy/setup.sh — установка окружения на одну VM (сервер или клиент)
# Запускать: bash ~/ddl/deploy/setup.sh
#
# Целевое окружение: Ubuntu 24.04, Python 3.12 (системный)
# Пакеты: flwr 1.28.0, torch 2.11.0 (CPU), torchvision 0.26.0, timm 1.0.26
# =============================================================================
set -euo pipefail

REPO_DIR="$HOME/ddl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -----------------------------------------------------------------------------
# 1. Системные пакеты
# -----------------------------------------------------------------------------
NEED_APT=false
command -v python3.12 &>/dev/null || NEED_APT=true
python3.12 -c "import ensurepip" &>/dev/null 2>&1 || NEED_APT=true
command -v tmux &>/dev/null || NEED_APT=true

if $NEED_APT; then
  log "Installing system packages..."
  sudo apt-get update -q
  sudo apt-get install -y -q \
    python3.12 python3.12-venv python3.12-dev \
    tmux htop git curl ca-certificates
else
  log "System packages OK, skipping apt"
fi

# -----------------------------------------------------------------------------
# 2. Python окружение
# -----------------------------------------------------------------------------
cd "$REPO_DIR"

FLWR_OK=false
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  installed=$(python -c "import flwr; print(flwr.__version__)" 2>/dev/null || echo "")
  [[ "$installed" == "1.28.0" ]] && FLWR_OK=true
fi

if [[ "$FLWR_OK" == false ]]; then
  log "Creating/updating Python venv..."
  python3.12 -m venv .venv
  source .venv/bin/activate

  pip install --upgrade pip setuptools wheel -q

  log "Installing PyTorch 2.11.0 (CPU)..."
  pip install \
    torch==2.11.0+cpu \
    torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu -q

  log "Installing Flower and dependencies..."
  pip install \
    "flwr[simulation]==1.28.0" \
    "datasets>=4.8.0" \
    "timm>=1.0.0" \
    "numpy>=2.0" \
    "pandas>=2.0" \
    "pillow>=12.0" \
    "huggingface_hub>=1.0" \
    "safetensors" \
    "rich" \
    "tqdm" \
    -q
else
  log "venv with flwr==1.28.0 already present, skipping pip install"
fi

# -----------------------------------------------------------------------------
# 3. Итог
# -----------------------------------------------------------------------------
log "================================================="
log "Setup complete on $(hostname)"
log "Python: $(python --version)"
log "flwr:   $(python -c 'import flwr; print(flwr.__version__)')"
log "torch:  $(python -c 'import torch; print(torch.__version__)')"
log "================================================="
