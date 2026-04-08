#!/usr/bin/env bash
# ==============================================================================
# deploy/common.sh — общие переменные и функции для всех deploy-скриптов
# Не запускать напрямую. Использовать:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# ==============================================================================

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/nodes.conf"

# ── Построение массивов узлов ─────────────────────────────────────────────────
NODE_NAMES=()
NODE_HOSTS=()
NODE_PORTS=()
NODE_USERS=()
NODE_KEYS=()

_i=1
while true; do
    _h="NODE_${_i}_HOST"
    [[ -z "${!_h:-}" ]] && break
    NODE_HOSTS+=("${!_h}")
    _p="NODE_${_i}_PORT";  NODE_PORTS+=("${!_p:-22}")
    _u="NODE_${_i}_USER";  NODE_USERS+=("${!_u:-$SERVER_USER}")
    _k="NODE_${_i}_KEY";   NODE_KEYS+=("${!_k:-$SERVER_KEY}")
    _n="NODE_${_i}_NAME";  NODE_NAMES+=("${!_n:-node-$_i}")
    ((_i++))
done
unset _i _h _p _u _k _n

NUM_NODES=${#NODE_HOSTS[@]}

# ── SSH-ключ: развернуть ~ в абсолютный путь ─────────────────────────────────
_expand_key() {
    local key="$1"
    echo "${key/#\~/$HOME}"
}

# ── SSH-функции ───────────────────────────────────────────────────────────────

_ssh_opts() {
    local key="$(_expand_key "$1")"
    local port="${2:-22}"
    echo "-i $key -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
}

# ssh_server "command"
ssh_server() {
    local opts; opts=$(_ssh_opts "$SERVER_KEY" "$SERVER_PORT")
    ssh $opts "${SERVER_USER}@${SERVER_HOST}" "$@"
}

# ssh_node INDEX "command"   (INDEX 0-based)
ssh_node() {
    local idx="$1"; shift
    local opts; opts=$(_ssh_opts "${NODE_KEYS[$idx]}" "${NODE_PORTS[$idx]}")
    ssh $opts "${NODE_USERS[$idx]}@${NODE_HOSTS[$idx]}" "$@"
}

# scp_to_node INDEX local_path remote_path
scp_to_node() {
    local idx="$1"
    local src="$2"
    local dst="$3"
    local key; key=$(_expand_key "${NODE_KEYS[$idx]}")
    local port="${NODE_PORTS[$idx]}"
    scp -i "$key" -P "$port" \
        -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -r "$src" "${NODE_USERS[$idx]}@${NODE_HOSTS[$idx]}:$dst"
}

# scp_to_server local_path remote_path
scp_to_server() {
    local key; key=$(_expand_key "$SERVER_KEY")
    scp -i "$key" -P "$SERVER_PORT" \
        -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -r "$1" "${SERVER_USER}@${SERVER_HOST}:$2"
}

# ── Логирование ──────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] + $*"; }
fail() { echo "[$(date '+%H:%M:%S')] - $*" >&2; }
