#!/usr/bin/env bash
# ==============================================================================
# deploy/start_training.sh — запуск FL-обучения с локальной машины
#
# Использование:
#   bash deploy/start_training.sh              # запуск с текущим pyproject.toml
#   bash deploy/start_training.sh --skip-push  # без git push (код уже на сервере)
#
# Что делает:
#   1. git push + git pull на сервере
#   2. Запускает flower-superlink на сервере (tmux)
#   3. Запускает ddl-client на каждом клиенте (tmux)
#   4. Ждёт подключения клиентов
#   5. Запускает flwr run на сервере
#   6. Стримит логи обучения
# ==============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SKIP_PUSH=false
for arg in "$@"; do
    [[ "$arg" == "--skip-push" ]] && SKIP_PUSH=true
done

# Читаем конфиг из pyproject.toml
NUM_CLIENTS=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    cfg = tomllib.load(f)
c = cfg['tool']['flwr']['app']['config']
print(c.get('min-available-nodes', 5))
" 2>/dev/null)
NUM_CLIENTS=${NUM_CLIENTS:-5}

FEDERATION=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg['tool']['flwr']['federations'].get('default', 'remote'))
" 2>/dev/null)
FEDERATION=${FEDERATION:-remote}

echo "============================================"
echo "  Запуск FL-обучения"
echo "  Клиентов: $NUM_CLIENTS"
echo "  Федерация: $FEDERATION"
echo "============================================"
echo ""

# ── 1. Синхронизация кода ─────────────────────────────────────────────────────
if ! $SKIP_PUSH; then
    log "Пушим код..."
    git push 2>&1 | tail -1
    log "Пуллим на сервере..."
    ssh_server "cd ~/ddl && git pull" 2>&1 | tail -3
    echo ""
fi

# ── 2. Остановка старых процессов ─────────────────────────────────────────────
log "Останавливаю старые процессы..."

ssh_server "tmux kill-session -t superlink 2>/dev/null; pkill -f flower-superlink 2>/dev/null" 2>/dev/null
for ((i=0; i<NUM_CLIENTS; i++)); do
    ssh_node "$i" "tmux kill-session -t client 2>/dev/null; pkill -f flower-supernode 2>/dev/null" 2>/dev/null &
done
wait
ok "Старые процессы остановлены"
echo ""

# ── 3. Запуск SuperLink ──────────────────────────────────────────────────────
log "Запускаю SuperLink на сервере..."
ssh_server "tmux new-session -d -s superlink 'cd ~/ddl && source .venv/bin/activate && flower-superlink --insecure 2>&1 | tee /tmp/superlink.log'"
sleep 2

# Проверяем что SuperLink запустился
if ssh_server "tmux has-session -t superlink 2>/dev/null"; then
    ok "SuperLink запущен"
else
    fail "SuperLink не запустился!"
    exit 1
fi
echo ""

# ── 4. Запуск клиентов ────────────────────────────────────────────────────────
log "Запускаю клиентов..."
for ((i=0; i<NUM_CLIENTS; i++)); do
    name="${NODE_NAMES[$i]}"
    log "  [$name] partition-id=$i"
    ssh_node "$i" "tmux new-session -d -s client 'cd ~/ddl && source .venv/bin/activate && ddl-client start --server ${SERVER_HOST}:${FLOWER_FLEET_PORT} --partition-id $i 2>&1 | tee /tmp/client.log'"
done

# Ждём подключения клиентов
log "Ожидаю подключения клиентов..."
TIMEOUT=60
CONNECTED=0
for ((t=0; t<TIMEOUT; t+=3)); do
    sleep 3
    # Считаем подключённых клиентов через лог SuperLink
    CONNECTED=$(ssh_server "grep -c 'Activate' /tmp/superlink.log 2>/dev/null" 2>/dev/null || echo "0")
    echo -ne "\r  Подключено: $CONNECTED/$NUM_CLIENTS (${t}s)..."
    if [[ "$CONNECTED" -ge "$NUM_CLIENTS" ]]; then
        break
    fi
done
echo ""

if [[ "$CONNECTED" -lt "$NUM_CLIENTS" ]]; then
    fail "Подключилось только $CONNECTED/$NUM_CLIENTS клиентов за ${TIMEOUT}s"
    echo "Проверьте логи клиентов: ssh <client> 'cat /tmp/client.log'"
    echo ""
    echo "Продолжить запуск? (y/n)"
    read -r ans
    [[ "$ans" != "y" ]] && exit 1
fi
ok "Все клиенты подключены"
echo ""

# ── 5. Запуск flwr run ───────────────────────────────────────────────────────
log "Запускаю flwr run..."
# flwr run на сервере в tmux, логи в файл
ssh_server "tmux new-session -d -s flwr-run 'cd ~/ddl && source .venv/bin/activate && flwr run . $FEDERATION 2>&1 | tee /tmp/flwr-run.log'"

# Ждём появления run-id
sleep 3
RUN_ID=$(ssh_server "grep -oP 'run\s+\K\d+' /tmp/flwr-run.log 2>/dev/null | head -1" 2>/dev/null)

if [[ -n "$RUN_ID" ]]; then
    ok "Run запущен: ID=$RUN_ID"
else
    log "Run-id ещё не появился, ждём..."
    sleep 5
    RUN_ID=$(ssh_server "grep -oP 'run\s+\K\d+' /tmp/flwr-run.log 2>/dev/null | head -1" 2>/dev/null)
    if [[ -n "$RUN_ID" ]]; then
        ok "Run запущен: ID=$RUN_ID"
    else
        fail "Не удалось получить run-id. Проверьте: ssh server 'cat /tmp/flwr-run.log'"
        exit 1
    fi
fi
echo ""

# ── 6. Стриминг логов ────────────────────────────────────────────────────────
echo "============================================"
echo "  Логи обучения (Ctrl+C для выхода)"
echo "============================================"
echo ""

ssh_server "cd ~/ddl && source .venv/bin/activate && flwr log $RUN_ID . $FEDERATION 2>&1"
