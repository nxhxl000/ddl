#!/usr/bin/env bash
# ==============================================================================
# deploy/stop_all.sh — останавливает все Flower процессы на всех VM
# ==============================================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STOP_CMD="
  tmux kill-session -t superlink 2>/dev/null && echo 'superlink stopped' || true
  tmux kill-session -t supernode 2>/dev/null && echo 'supernode stopped' || true
  pkill -f 'flower-superlink' 2>/dev/null || true
  pkill -f 'flower-supernode' 2>/dev/null || true
  echo 'Done on \$(hostname)'
"

log "Stopping all Flower processes..."

ssh_server "$STOP_CMD" 2>&1 | sed "s/^/[fl-server] /" &

for IDX in "${!CLIENT_INT_IPS[@]}"; do
  name="${CLIENT_NAMES[$IDX]}"
  ssh_client "${CLIENT_INT_IPS[$IDX]}" "$STOP_CMD" 2>&1 | sed "s/^/[$name] /" &
done

wait
log "All stopped."
