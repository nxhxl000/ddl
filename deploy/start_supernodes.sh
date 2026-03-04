#!/usr/bin/env bash
# ==============================================================================
# deploy/start_supernodes.sh — запуск SuperNode на всех клиентских VM
# Запускать локально с твоего ноутбука.
# ==============================================================================

# Намеренно БЕЗ set -e: падение SSH к одному клиенту не должно
# останавливать запуск остальных

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SESSION="supernode"
NUM_PARTITIONS="${#CLIENT_INT_IPS[@]}"

log "Starting $NUM_PARTITIONS SuperNodes (partition-id 0...$((NUM_PARTITIONS-1)))..."

for IDX in "${!CLIENT_INT_IPS[@]}"; do
  int_ip="${CLIENT_INT_IPS[$IDX]}"
  name="${CLIENT_NAMES[$IDX]}"
  log "  → $name ($int_ip) partition-id=$IDX"

  ssh_client "$int_ip" "
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION \
      'cd $REMOTE_DIR && source .venv/bin/activate && flower-supernode --superlink $SERVER_INT:9092 --insecure --node-config \"partition-id=$IDX\" --node-config \"num-partitions=$NUM_PARTITIONS\"' \
      Enter
    echo 'SuperNode partition-id=$IDX started on '\$(hostname)
  " && log "  ✓ $name OK" \
    || log "  ✗ $name FAILED"
done

log "Waiting 4s for SuperNodes to connect..."
sleep 4

log "Status (tmux sessions):"
for IDX in "${!CLIENT_INT_IPS[@]}"; do
  int_ip="${CLIENT_INT_IPS[$IDX]}"
  name="${CLIENT_NAMES[$IDX]}"
  RESULT=$(ssh_client "$int_ip" \
    "tmux list-sessions 2>/dev/null | grep $SESSION || echo 'NO SESSION'" 2>/dev/null \
    || echo "SSH FAILED")
  echo "  $name: $RESULT"
done
