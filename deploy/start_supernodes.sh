#!/usr/bin/env bash
# ==============================================================================
# deploy/start_supernodes.sh — запуск SuperNode на всех клиентских VM
# Запускать локально.
# ==============================================================================

# Намеренно БЕЗ set -e: падение SSH к одному клиенту не должно
# останавливать запуск остальных

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SESSION="supernode"

log "Starting $NUM_NODES SuperNodes (partition-id 0...$((NUM_NODES-1)))..."

for ((i=0; i<NUM_NODES; i++)); do
  name="${NODE_NAMES[$i]}"
  log "  → $name (${NODE_HOSTS[$i]}) partition-id=$i"

  ssh_node "$i" "
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION \
      \"bash -c 'cd \$HOME/$REMOTE_DIR && source .venv/bin/activate && exec flower-supernode --superlink ${SERVER_HOST}:${FLOWER_FLEET_PORT} --insecure --node-config \\\"partition-id=$i num-partitions=$NUM_NODES\\\"'\"
    echo 'SuperNode partition-id=$i started on '\$(hostname)
  " && log "  + $name OK" \
    || log "  - $name FAILED"
done

log "Waiting 5s for SuperNodes to connect..."
sleep 5

log "Status (tmux sessions):"
for ((i=0; i<NUM_NODES; i++)); do
  name="${NODE_NAMES[$i]}"
  RESULT=$(ssh_node "$i" \
    "tmux list-sessions 2>/dev/null | grep $SESSION || echo 'NO SESSION'" 2>/dev/null \
    || echo "SSH FAILED")
  echo "  $name: $RESULT"
done
