#!/usr/bin/env bash
# =============================================================================
# deploy/start_supernodes.sh — запуск SuperNode на всех клиентских VM
# Запускать локально с твоего ноутбука.
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ — должна совпадать с deploy_all.sh
# -----------------------------------------------------------------------------
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
NUM_PARTITIONS="${#CLIENT_IPS[@]}"
SESSION="supernode"
# -----------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

setup_ssh_key() {
  if [[ ! -f "$SSH_KEY" ]] || ! diff -q "$SSH_KEY_WIN" "$SSH_KEY" &>/dev/null; then
    mkdir -p "$HOME/.ssh"
    cp "$SSH_KEY_WIN" "$SSH_KEY"
    chmod 600 "$SSH_KEY"
    log "SSH key copied to $SSH_KEY"
  fi
}

check_config() {
  if [[ -z "$SERVER_INT" ]]; then
    echo "ERROR: Заполни SERVER_INT в deploy/start_supernodes.sh"
    exit 1
  fi
  for ip in "${CLIENT_IPS[@]}"; do
    if [[ -z "$ip" ]]; then
      echo "ERROR: Заполни все CLIENT_IPS в deploy/start_supernodes.sh"
      exit 1
    fi
  done
}

ssh_cmd() {
  local ip="$1"; shift
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$ip" "$@"
}

setup_ssh_key
check_config

log "Starting $NUM_PARTITIONS SuperNodes (partition-id 0...$((NUM_PARTITIONS-1)))..."

PARTITION=0
for ip in "${CLIENT_IPS[@]}"; do
  ssh_cmd "$ip" "
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION
    tmux send-keys -t $SESSION \
      'cd $REMOTE_DIR && source .venv/bin/activate && \
       flower-supernode \
         --superlink $SERVER_INT:9092 \
         --insecure \
         --node-config \"partition-id=$PARTITION,num-partitions=$NUM_PARTITIONS\"' \
      Enter
    echo \"SuperNode partition-id=$PARTITION started on \$(hostname)\"
  " &
  ((PARTITION++))
done

wait

log "All SuperNodes started. Checking connectivity in 3s..."
sleep 3

log "Status:"
PARTITION=0
for ip in "${CLIENT_IPS[@]}"; do
  STATUS=$(ssh_cmd "$ip" "pgrep -a python | grep supernode | head -1 || echo 'NOT RUNNING'")
  echo "  client-$PARTITION ($ip): $STATUS"
  ((PARTITION++))
done
