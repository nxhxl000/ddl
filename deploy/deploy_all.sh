#!/usr/bin/env bash
# =============================================================================
# deploy/deploy_all.sh — деплой кода и запуск setup.sh на всех VM
#
# Использование:
#   bash deploy/deploy_all.sh            # полный деплой
#   bash deploy/deploy_all.sh --sync-only  # только rsync, без setup
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

SSH_KEY_WIN="/mnt/c/Users/listr/.ssh/admin-fl"
SSH_KEY="$HOME/.ssh/admin-fl"
SSH_USER="gleb"
REMOTE_DIR="~/ddl"

RSYNC_TIMEOUT=60    # сек на один rsync (если завис — убиваем)
SSH_TIMEOUT=15      # сек на установку SSH соединения
# -----------------------------------------------------------------------------

SYNC_ONLY=false
for arg in "$@"; do [[ "$arg" == "--sync-only" ]] && SYNC_ONLY=true; done

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
fail() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }

setup_ssh_key() {
  if [[ ! -f "$SSH_KEY" ]] || ! diff -q "$SSH_KEY_WIN" "$SSH_KEY" &>/dev/null; then
    mkdir -p "$HOME/.ssh"
    cp "$SSH_KEY_WIN" "$SSH_KEY"
    chmod 600 "$SSH_KEY"
    log "SSH key copied to $SSH_KEY"
  fi
}

# SSH и rsync с таймаутами — зависшее соединение не заблокирует скрипт
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=$SSH_TIMEOUT -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

ssh_cmd() {
  local ip="$1"; shift
  ssh $SSH_OPTS "$SSH_USER@$ip" "$@"
}

rsync_to() {
  local ip="$1"
  timeout "$RSYNC_TIMEOUT" rsync -az \
    -e "ssh $SSH_OPTS" \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'data/' \
    --exclude 'runs/' \
    --exclude 'local_exp/' \
    --exclude 'models/' \
    ./ "$SSH_USER@$ip:$REMOTE_DIR/" \
  && ok "rsync → $ip" \
  || { fail "rsync → $ip (timeout или ошибка)"; return 1; }
}

# -----------------------------------------------------------------------------
setup_ssh_key

ALL_IPS=("$SERVER_EXT" "${CLIENT_IPS[@]}")

# 1. Rsync — параллельно с timeout
log "Syncing code to all VMs (parallel, timeout ${RSYNC_TIMEOUT}s each)..."
FAILED_RSYNC=()
for ip in "${ALL_IPS[@]}"; do
  rsync_to "$ip" &
done
# Ждём все фоновые rsync, собираем ошибки вручную (set -e не ловит фоновые)
RSYNC_FAIL=0
wait || RSYNC_FAIL=1

if [[ $RSYNC_FAIL -ne 0 ]]; then
  fail "Один или несколько rsync завершились с ошибкой — проверь вывод выше"
  fail "Продолжаем (остальные VM могут быть в порядке)"
fi
log "Rsync phase done."

[[ "$SYNC_ONLY" == true ]] && { log "--sync-only: stopping here."; exit 0; }

# 2. Setup — последовательно, с чётким выводом и меткой VM
# Сервер без датасетов, клиенты с датасетами
log "Running setup on SERVER..."
ssh_cmd "$SERVER_EXT" "bash $REMOTE_DIR/deploy/setup.sh --skip-data" \
  2>&1 | sed "s/^/[server] /" \
  && ok "Server setup done" \
  || fail "Server setup FAILED"

IDX=1
for ip in "${CLIENT_IPS[@]}"; do
  log "Running setup on client-$IDX ($ip)..."
  ssh_cmd "$ip" "bash $REMOTE_DIR/deploy/setup.sh" \
    2>&1 | sed "s/^/[client-$IDX] /" \
    && ok "client-$IDX setup done" \
    || fail "client-$IDX setup FAILED"
  ((IDX++))
done

# 3. Добавляем секцию yandex-cloud в pyproject.toml на сервере (если нет)
log "Patching pyproject.toml on server..."
ssh_cmd "$SERVER_EXT" "
  grep -q 'yandex-cloud' $REMOTE_DIR/pyproject.toml && echo 'already patched' || \
  printf '\n[tool.flwr.federations.yandex-cloud]\naddress = \"${SERVER_EXT}:9093\"\ninsecure = true\n' \
    >> $REMOTE_DIR/pyproject.toml && echo 'patched'
"

log "================================================="
log "Deploy complete!"
log ""
log "Next steps:"
log "  1. ssh -i ~/.ssh/admin-fl gleb@$SERVER_EXT 'bash ~/ddl/deploy/start_superlink.sh'"
log "  2. bash deploy/start_supernodes.sh"
log "  3. flwr run . yandex-cloud"
log "================================================="
