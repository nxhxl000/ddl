#!/usr/bin/env bash
# ==============================================================================
# deploy/deploy_all.sh — деплой на сервер и все клиентские VM
#
# FAB (Flower Application Bundle) доставляет код автоматически через SuperLink.
# Этот скрипт деплоит только:
#   - на сервер: код проекта + pyproject.toml (для SuperLink + flwr run)
#   - на клиентов: deploy/setup.sh (для установки зависимостей)
#
# Данные (партиции) деплоятся отдельно через deploy/deploy_data.sh или cluster.ipynb.
#
# Использование:
#   bash deploy/deploy_all.sh              # полный деплой
#   bash deploy/deploy_all.sh --sync-only  # только rsync, без setup
# ==============================================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SYNC_ONLY=false
for arg in "$@"; do [[ "$arg" == "--sync-only" ]] && SYNC_ONLY=true; done

RSYNC_TIMEOUT=120  # сек на один rsync

_expand_key() { echo "${1/#\~/$HOME}"; }

# ── Rsync-функции ────────────────────────────────────────────────────────────

rsync_to_server() {
  local key; key=$(_expand_key "$SERVER_KEY")
  local ssh_cmd="ssh -i $key -p $SERVER_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
  timeout "$RSYNC_TIMEOUT" rsync -az \
    -e "$ssh_cmd" \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'data/' --exclude 'runs/' \
    --exclude 'notebooks/' \
    ./ "${SERVER_USER}@${SERVER_HOST}:\$HOME/$REMOTE_DIR/" \
  && ok "rsync → server" \
  || { fail "rsync → server FAILED"; return 1; }
}

rsync_to_node() {
  local idx="$1"
  local name="${NODE_NAMES[$idx]}"
  local host="${NODE_HOSTS[$idx]}"
  local port="${NODE_PORTS[$idx]}"
  local user="${NODE_USERS[$idx]}"
  local key; key=$(_expand_key "${NODE_KEYS[$idx]}")
  local ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=10"

  # На клиентов — только setup.sh и requirements для установки зависимостей.
  # Код приложения доставляется через FAB автоматически.
  timeout "$RSYNC_TIMEOUT" rsync -az \
    -e "$ssh_cmd" \
    --include 'deploy/' --include 'deploy/setup.sh' --include 'deploy/common.sh' \
    --include 'deploy/nodes.conf' \
    --include 'requirements.txt' --include 'pyproject.toml' \
    --exclude '*' \
    ./ "${user}@${host}:\$HOME/$REMOTE_DIR/" \
  && ok "rsync → $name" \
  || { fail "rsync → $name FAILED"; return 1; }
}

# ── 1. Rsync — параллельно ───────────────────────────────────────────────────
log "Syncing to server + $NUM_NODES clients (parallel, timeout ${RSYNC_TIMEOUT}s each)..."

rsync_to_server &
for ((i=0; i<NUM_NODES; i++)); do
  rsync_to_node "$i" &
done

RSYNC_FAIL=0
wait || RSYNC_FAIL=1

if [[ $RSYNC_FAIL -ne 0 ]]; then
  fail "Один или несколько rsync завершились с ошибкой — проверь вывод выше"
  fail "Продолжаем (остальные VM могут быть в порядке)"
fi
log "Rsync phase done."

[[ "$SYNC_ONLY" == true ]] && { log "--sync-only: stopping here."; exit 0; }

# ── 2. Setup — последовательно ───────────────────────────────────────────────
log "Running setup on server..."
ssh_server "bash \$HOME/$REMOTE_DIR/deploy/setup.sh" \
  2>&1 | sed "s/^/[server] /" \
  && ok "server setup done" \
  || fail "server setup FAILED"

for ((i=0; i<NUM_NODES; i++)); do
  name="${NODE_NAMES[$i]}"
  log "Running setup on $name..."
  ssh_node "$i" "bash \$HOME/$REMOTE_DIR/deploy/setup.sh" \
    2>&1 | sed "s/^/[$name] /" \
    && ok "$name setup done" \
    || fail "$name setup FAILED"
done

# ── 3. Итог ──────────────────────────────────────────────────────────────────
log "================================================="
log "Deploy complete! Nodes: server + $NUM_NODES clients"
log ""
log "Next steps:"
log "  1. bash deploy/start_superlink.sh  (на сервере)"
log "  2. bash deploy/start_supernodes.sh (локально)"
log "  3. flwr run . remote"
log "================================================="
