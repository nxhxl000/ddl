#!/usr/bin/env bash
# ==============================================================================
# deploy/deploy_all.sh — деплой кода и запуск setup.sh на всех VM
#
# Использование:
#   bash deploy/deploy_all.sh            # полный деплой
#   bash deploy/deploy_all.sh --sync-only  # только rsync, без setup
# ==============================================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SYNC_ONLY=false
for arg in "$@"; do [[ "$arg" == "--sync-only" ]] && SYNC_ONLY=true; done

RSYNC_TIMEOUT=60  # сек на один rsync

rsync_to_server() {
  timeout "$RSYNC_TIMEOUT" rsync -az \
    -e "ssh $SSH_OPTS" \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.git' --exclude 'data/' --exclude 'runs/' \
    --exclude 'local_exp/' --exclude 'models/' \
    ./ "$SSH_USER@$SERVER_EXT:$REMOTE_DIR/" \
  && ok "rsync → fl-server" \
  || { fail "rsync → fl-server FAILED"; return 1; }
}

rsync_to_client() {
  local name="$1" int_ip="$2"
  local proxy="ssh $SSH_OPTS -W %h:%p $SSH_USER@$SERVER_EXT"
  timeout "$RSYNC_TIMEOUT" rsync -az \
    -e "ssh $SSH_OPTS -o 'ProxyCommand=$proxy'" \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.git' --exclude 'data/' --exclude 'runs/' \
    --exclude 'local_exp/' --exclude 'models/' \
    ./ "$SSH_USER@$int_ip:$REMOTE_DIR/" \
  && ok "rsync → $name" \
  || { fail "rsync → $name FAILED"; return 1; }
}

# 1. Rsync — параллельно
log "Syncing code to all VMs (parallel, timeout ${RSYNC_TIMEOUT}s each)..."
rsync_to_server &
for IDX in "${!CLIENT_INT_IPS[@]}"; do
  rsync_to_client "${CLIENT_NAMES[$IDX]}" "${CLIENT_INT_IPS[$IDX]}" &
done
RSYNC_FAIL=0
wait || RSYNC_FAIL=1

if [[ $RSYNC_FAIL -ne 0 ]]; then
  fail "Один или несколько rsync завершились с ошибкой — проверь вывод выше"
  fail "Продолжаем (остальные VM могут быть в порядке)"
fi
log "Rsync phase done."

[[ "$SYNC_ONLY" == true ]] && { log "--sync-only: stopping here."; exit 0; }

# 2. Setup — последовательно (сервер без датасетов, клиенты с датасетами)
log "Running setup on fl-server..."
ssh_server "bash $REMOTE_DIR/deploy/setup.sh --skip-data" \
  2>&1 | sed "s/^/[fl-server] /" \
  && ok "fl-server setup done" \
  || fail "fl-server setup FAILED"

for IDX in "${!CLIENT_INT_IPS[@]}"; do
  name="${CLIENT_NAMES[$IDX]}"
  int_ip="${CLIENT_INT_IPS[$IDX]}"
  log "Running setup on $name..."
  ssh_client "$int_ip" "bash $REMOTE_DIR/deploy/setup.sh" \
    2>&1 | sed "s/^/[$name] /" \
    && ok "$name setup done" \
    || fail "$name setup FAILED"
done

# 3. Патчим pyproject.toml на сервере (добавляем секцию yandex-cloud если нет)
log "Patching pyproject.toml on server..."
ssh_server "
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
