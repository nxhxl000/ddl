#!/usr/bin/env bash
# =============================================================================
# deploy/stop_all.sh — останавливает все Flower процессы на всех VM
# =============================================================================
set -euo pipefail

# Конфигурация (совпадает с остальными скриптами)
SERVER_EXT="84.201.165.255"
CLIENT_IPS=(
  "89.169.162.84"
  "89.169.183.99"
  "84.201.179.134"
  "62.84.120.239"
  "89.169.166.235"
)

SSH_KEY_WIN="/mnt/c/Users/listr/.ssh/admin-fl"
SSH_KEY="$HOME/.ssh/admin-fl"
SSH_USER="gleb"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_cmd() { ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$1" "${@:2}"; }

# Копируем ключ из Windows в WSL с правами 600
if [[ ! -f "$SSH_KEY" ]] || ! diff -q "$SSH_KEY_WIN" "$SSH_KEY" &>/dev/null; then
  mkdir -p "$HOME/.ssh" && cp "$SSH_KEY_WIN" "$SSH_KEY" && chmod 600 "$SSH_KEY"
fi

ALL_IPS=("$SERVER_EXT" "${CLIENT_IPS[@]}")

log "Stopping all Flower processes..."
for ip in "${ALL_IPS[@]}"; do
  [[ -z "$ip" ]] && continue
  ssh_cmd "$ip" "
    tmux kill-session -t superlink 2>/dev/null && echo 'superlink stopped' || true
    tmux kill-session -t supernode 2>/dev/null && echo 'supernode stopped' || true
    pkill -f 'flwr superlink' 2>/dev/null || true
    pkill -f 'flwr supernode' 2>/dev/null || true
    echo 'Done on \$(hostname)'
  " 2>&1 | sed "s/^/[$ip] /" &
done
wait
log "All stopped."
