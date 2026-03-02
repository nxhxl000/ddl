#!/usr/bin/env bash
# =============================================================================
# deploy/start_superlink.sh — запуск SuperLink на сервере
# Запускать на самом сервере: bash ~/ddl/deploy/start_superlink.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$HOME/ddl"
SESSION="superlink"

cd "$REPO_DIR"
source .venv/bin/activate

# Убиваем старую сессию если есть
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" \
  "cd $REPO_DIR && source .venv/bin/activate && flower-superlink --insecure" Enter

echo "SuperLink started in tmux session '$SESSION'"
echo "Attach: tmux attach -t $SESSION"
echo "Ports:  9092 (Fleet API for SuperNodes)"
echo "        9093 (Exec API for flwr run)"
