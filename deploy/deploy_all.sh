#!/usr/bin/env bash
# =============================================================================
# deploy/deploy_all.sh — деплой кода и запуск setup.sh на всех VM
#
# Использование:
#   bash deploy/deploy_all.sh
#
# Перед запуском заполни IP адреса ниже.
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# -----------------------------------------------------------------------------
SERVER_EXT="84.201.165.255"
SERVER_INT="10.10.0.30"

CLIENT_IPS=(
  "89.169.162.84"    # fl-client1
  "89.169.183.99"    # fl-client2
  "84.201.179.134"   # fl-client3
  "62.84.120.239"    # fl-client4
  "89.169.166.235"   # fl-client5
)

SSH_KEY_WIN="/mnt/c/Users/listr/.ssh/admin-fl"   # путь на Windows (через WSL)
SSH_KEY="$HOME/.ssh/admin-fl"                     # куда копируем для SSH
SSH_USER="gleb"
REMOTE_DIR="~/ddl"
# -----------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# SSH в WSL не принимает ключи с /mnt/c/ (NTFS не поддерживает chmod 600)
# Копируем ключ в ~/.ssh/ с правильными правами — один раз
setup_ssh_key() {
  if [[ ! -f "$SSH_KEY" ]] || ! diff -q "$SSH_KEY_WIN" "$SSH_KEY" &>/dev/null; then
    mkdir -p "$HOME/.ssh"
    cp "$SSH_KEY_WIN" "$SSH_KEY"
    chmod 600 "$SSH_KEY"
    log "SSH key copied to $SSH_KEY"
  fi
}

check_config() {
  if [[ -z "$SERVER_EXT" || -z "$SERVER_INT" ]]; then
    echo "ERROR: Заполни SERVER_EXT и SERVER_INT в deploy/deploy_all.sh"
    exit 1
  fi
  for ip in "${CLIENT_IPS[@]}"; do
    if [[ -z "$ip" ]]; then
      echo "ERROR: Заполни все CLIENT_IPS в deploy/deploy_all.sh"
      exit 1
    fi
  done
}

ssh_cmd() {
  local ip="$1"; shift
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$ip" "$@"
}

rsync_to() {
  local ip="$1"
  rsync -avz --progress \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'data/' \
    --exclude 'runs/' \
    --exclude 'local_exp/' \
    --exclude 'models/' \
    ./ "$SSH_USER@$ip:$REMOTE_DIR/"
}

# -----------------------------------------------------------------------------
setup_ssh_key
check_config

ALL_IPS=("$SERVER_EXT" "${CLIENT_IPS[@]}")

# 1. Синхронизируем код
log "Syncing code to all VMs..."
for ip in "${ALL_IPS[@]}"; do
  rsync_to "$ip" &
done
wait
log "Code synced."

# 2. Запускаем setup.sh параллельно
log "Running setup.sh on all VMs..."

ssh_cmd "$SERVER_EXT" "bash $REMOTE_DIR/deploy/setup.sh --skip-data" \
  2>&1 | sed "s/^/[server] /" &

PARTITION=0
for ip in "${CLIENT_IPS[@]}"; do
  ssh_cmd "$ip" "bash $REMOTE_DIR/deploy/setup.sh" \
    2>&1 | sed "s/^/[client-$PARTITION] /" &
  ((PARTITION++))
done

wait
log "All VMs set up."

# 3. Обновляем pyproject.toml с реальным IP сервера
log "Patching pyproject.toml with server address..."
TOML_REMOTE="$REMOTE_DIR/pyproject.toml"

# Добавляем секцию yandex-cloud если её нет
ssh_cmd "$SERVER_EXT" "
  grep -q 'yandex-cloud' $TOML_REMOTE || cat >> $TOML_REMOTE << 'EOF'

[tool.flwr.federations.yandex-cloud]
address = \"${SERVER_EXT}:9093\"
insecure = true
EOF
  echo 'pyproject.toml updated on server'
"

log "================================================="
log "Deploy complete!"
log "Next steps:"
log "  1. Start SuperLink on server:"
log "     ssh $SSH_USER@$SERVER_EXT"
log "     bash $REMOTE_DIR/deploy/start_superlink.sh"
log ""
log "  2. Start SuperNodes on clients:"
log "     bash deploy/start_supernodes.sh"
log ""
log "  3. Run federated learning:"
log "     flwr run . yandex-cloud"
log "================================================="
