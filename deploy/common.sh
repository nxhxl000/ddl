#!/usr/bin/env bash
# ==============================================================================
# deploy/common.sh — общие переменные и функции для всех deploy-скриптов
# Не запускать напрямую. Использовать:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# ==============================================================================

# Загружаем конфиг узлов (nodes.conf лежит рядом с этим файлом)
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/nodes.conf"

# Строим массивы CLIENT_NAMES / CLIENT_IPS (внешние) / CLIENT_INT_IPS (внутренние)
CLIENT_NAMES=()
CLIENT_IPS=()
CLIENT_INT_IPS=()
_i=1
while true; do
  _v="CLIENT_$_i"; _ip="${!_v}"
  [[ -z "$_ip" ]] && break
  CLIENT_IPS+=("$_ip")
  _vi="CLIENT_${_i}_INT";  CLIENT_INT_IPS+=("${!_vi:-$_ip}")
  _vn="CLIENT_${_i}_NAME"; CLIENT_NAMES+=("${!_vn:-client-$_i}")
  ((_i++))
done
unset _i _v _vi _vn _ip

# SSH-ключ (WSL путь): копируем из Windows, если нужно
SSH_KEY="$HOME/.ssh/admin-fl"
if [[ ! -f "$SSH_KEY" ]] || ! diff -q "$SSH_KEY_WIN" "$SSH_KEY" &>/dev/null; then
  mkdir -p "$HOME/.ssh"
  cp "$SSH_KEY_WIN" "$SSH_KEY"
  chmod 600 "$SSH_KEY"
fi

# Базовые SSH-опции (без ProxyJump)
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

# Подключение к серверу напрямую
ssh_server() {
  ssh $SSH_OPTS "$SSH_USER@$SERVER_EXT" "$@"
}

# Подключение к клиенту через сервер как jump host (внутренний IP).
# Используем явный ProxyCommand — иначе OpenSSH не передаёт -i в jump-соединение.
ssh_client() {
  local int_ip="$1"; shift
  ssh $SSH_OPTS \
    -o "ProxyCommand=ssh $SSH_OPTS -W %h:%p $SSH_USER@$SERVER_EXT" \
    "$SSH_USER@$int_ip" "$@"
}

# Вспомогательные функции
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
fail() { echo "[$(date '+%H:%M:%S')] ✗ $*" >&2; }
