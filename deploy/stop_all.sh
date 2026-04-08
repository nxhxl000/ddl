#!/usr/bin/env bash
# ==============================================================================
# deploy/stop_all.sh — остановка всех FL-процессов на сервере и клиентах
# ==============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Останавливаю все FL-процессы..."

# Сервер
log "Сервер..."
ssh_server "tmux kill-session -t superlink 2>/dev/null; tmux kill-session -t flwr-run 2>/dev/null; pkill -f flower-superlink 2>/dev/null" 2>/dev/null
ok "Сервер"

# Клиенты (параллельно)
for ((i=0; i<NUM_NODES; i++)); do
    name="${NODE_NAMES[$i]}"
    log "$name..."
    ssh_node "$i" "tmux kill-session -t client 2>/dev/null; pkill -f flower-supernode 2>/dev/null" 2>/dev/null &
done
wait

ok "Все процессы остановлены"
