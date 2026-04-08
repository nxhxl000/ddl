#!/usr/bin/env bash
# ==============================================================================
# deploy/deploy_data.sh — деплой партиций на сервер и клиенты
#
# Использование:
#   bash deploy/deploy_data.sh <partition-name>
#
# Пример:
#   bash deploy/deploy_data.sh cifar100__iid__n5__s42
#
# Что делает:
#   Сервер  ← test/, manifest.json, server/ (если есть)
#   Клиент i ← client_{i}/, server/ (если есть)
#
# Если данные уже есть на узле (manifest.json существует) — пропускает.
# Для принудительной перезаливки: --force
# ==============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ── Аргументы ────────────────────────────────────────────────────────────────
FORCE=false
PARTITION_NAME=""

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        *)       PARTITION_NAME="$arg" ;;
    esac
done

if [[ -z "$PARTITION_NAME" ]]; then
    echo "Использование: bash deploy/deploy_data.sh <partition-name> [--force]"
    echo ""
    echo "Доступные партиции:"
    ls -1 data/partitions/ 2>/dev/null || echo "  (нет партиций в data/partitions/)"
    exit 1
fi

LOCAL_DIR="data/partitions/$PARTITION_NAME"

if [[ ! -d "$LOCAL_DIR" ]]; then
    fail "Локальная партиция не найдена: $LOCAL_DIR"
    exit 1
fi

if [[ ! -f "$LOCAL_DIR/manifest.json" ]]; then
    fail "manifest.json не найден в $LOCAL_DIR"
    exit 1
fi

# Читаем число клиентов из манифеста
NUM_CLIENTS=$(python3 -c "import json; print(json.load(open('$LOCAL_DIR/manifest.json'))['num_clients'])" 2>/dev/null)
if [[ -z "$NUM_CLIENTS" ]]; then
    fail "Не удалось прочитать num_clients из manifest.json"
    exit 1
fi

HAS_SERVER_DS=false
[[ -d "$LOCAL_DIR/server" ]] && HAS_SERVER_DS=true

# Относительный путь на удалённой машине (от $HOME)
REMOTE_PART="$REMOTE_DIR/data/partitions/$PARTITION_NAME"

echo "============================================"
echo "  Деплой партиции: $PARTITION_NAME"
echo "  Клиентов в партиции: $NUM_CLIENTS"
echo "  Узлов в кластере: $NUM_NODES"
echo "  Серверный датасет: $HAS_SERVER_DS"
echo "  Force: $FORCE"
echo "============================================"
echo ""

if [[ "$NUM_CLIENTS" -gt "$NUM_NODES" ]]; then
    fail "Партиция на $NUM_CLIENTS клиентов, но в кластере только $NUM_NODES узлов"
    exit 1
fi

# ── Сервер ────────────────────────────────────────────────────────────────────
echo "--- [server] ${SERVER_HOST}:${SERVER_PORT} ---"

server_has_data=false
if ssh_server "test -f \$HOME/$REMOTE_PART/manifest.json" 2>/dev/null; then
    server_has_data=true
fi

if $server_has_data && ! $FORCE; then
    ok "Данные уже есть, пропускаем"
else
    log "Создаю директорию..."
    ssh_server "mkdir -p \$HOME/$REMOTE_PART"

    log "Загружаю test/ и manifest.json..."
    scp_to_server "$LOCAL_DIR/manifest.json" "$REMOTE_PART/manifest.json"
    scp_to_server "$LOCAL_DIR/test/" "$REMOTE_PART/"

    if $HAS_SERVER_DS; then
        log "Загружаю server/..."
        scp_to_server "$LOCAL_DIR/server/" "$REMOTE_PART/"
    fi

    ok "Сервер готов"
fi
echo ""

# ── Клиенты ──────────────────────────────────────────────────────────────────
for ((i=0; i<NUM_CLIENTS; i++)); do
    name="${NODE_NAMES[$i]}"
    host="${NODE_HOSTS[$i]}"
    port="${NODE_PORTS[$i]}"
    echo "--- [$name] $host:$port → client_$i ---"

    node_has_data=false
    if ssh_node "$i" "test -f \$HOME/$REMOTE_PART/manifest.json" 2>/dev/null; then
        node_has_data=true
    fi

    if $node_has_data && ! $FORCE; then
        ok "Данные уже есть, пропускаем"
        echo ""
        continue
    fi

    log "Создаю директорию..."
    ssh_node "$i" "mkdir -p \$HOME/$REMOTE_PART"

    client_dir="$LOCAL_DIR/client_$i"
    if [[ ! -d "$client_dir" ]]; then
        fail "Локальная директория не найдена: $client_dir"
        echo ""
        continue
    fi

    log "Загружаю client_$i/..."
    scp_to_node "$i" "$client_dir/" "$REMOTE_PART/"

    scp_to_node "$i" "$LOCAL_DIR/manifest.json" "$REMOTE_PART/manifest.json"

    if $HAS_SERVER_DS; then
        log "Загружаю server/..."
        scp_to_node "$i" "$LOCAL_DIR/server/" "$REMOTE_PART/"
    fi

    ok "$name готов"
    echo ""
done

echo "============================================"
echo "  Деплой завершён: $PARTITION_NAME"
echo "============================================"
