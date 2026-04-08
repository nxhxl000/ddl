#!/usr/bin/env bash
# ==============================================================================
# deploy/check_clients.sh — проверка развёртывания на всех клиентах
#
# Проверяет:
#   1. SSH-подключение
#   2. Python 3.12
#   3. venv существует
#   4. ddl-client установлен и работает
#   5. Версии ключевых библиотек (torch, flwr, timm)
#
# Использование:
#   bash deploy/check_clients.sh          # все клиенты
#   bash deploy/check_clients.sh 1 3      # только node 1 и 3
# ==============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Какие узлы проверять (по умолчанию — все)
if [[ $# -gt 0 ]]; then
    INDICES=("$@")
else
    INDICES=($(seq 0 $((NUM_NODES - 1))))
fi

PASS=0
FAIL_COUNT=0
TOTAL=${#INDICES[@]}

echo "============================================"
echo "  Проверка клиентов ($TOTAL узлов)"
echo "============================================"
echo ""

for idx in "${INDICES[@]}"; do
    name="${NODE_NAMES[$idx]}"
    host="${NODE_HOSTS[$idx]}"
    port="${NODE_PORTS[$idx]}"
    echo "--- [$name] $host:$port ---"

    # 1. SSH
    if ! ssh_node "$idx" "echo ok" &>/dev/null; then
        fail "$name: SSH-подключение не удалось"
        ((FAIL_COUNT++))
        echo ""
        continue
    fi
    ok "SSH"

    # 2-5. Всё одной командой (экономим SSH-подключения)
    result=$(ssh_node "$idx" bash -s <<'REMOTE'
        errors=""

        # Python
        if command -v python3.12 &>/dev/null; then
            py_ver=$(python3.12 --version 2>&1)
            echo "PYTHON_OK:$py_ver"
        else
            echo "PYTHON_FAIL"
            errors="python "
        fi

        # venv
        if [ -d "$HOME/ddl/.venv" ]; then
            echo "VENV_OK"
        else
            echo "VENV_FAIL"
            errors="${errors}venv "
        fi

        # ddl-client
        source "$HOME/ddl/.venv/bin/activate" 2>/dev/null
        if command -v ddl-client &>/dev/null; then
            echo "CLIENT_OK"
        else
            echo "CLIENT_FAIL"
            errors="${errors}ddl-client "
        fi

        # Версии библиотек
        versions=$(python3.12 -c "
import importlib, json
libs = {}
for lib in ['torch', 'flwr', 'timm', 'torchvision', 'datasets']:
    try:
        m = importlib.import_module(lib)
        libs[lib] = m.__version__
    except:
        libs[lib] = 'NOT INSTALLED'
print(json.dumps(libs))
" 2>/dev/null)
        echo "VERSIONS:${versions:-{}}"

        # Диск
        disk=$(df -h "$HOME" 2>/dev/null | awk 'NR==2{print $4}')
        echo "DISK:$disk"

        echo "ERRORS:$errors"
REMOTE
    )

    # Парсим результаты
    py_line=$(echo "$result" | grep "^PYTHON_")
    venv_line=$(echo "$result" | grep "^VENV_")
    client_line=$(echo "$result" | grep "^CLIENT_")
    ver_line=$(echo "$result" | grep "^VERSIONS:")
    disk_line=$(echo "$result" | grep "^DISK:")
    err_line=$(echo "$result" | grep "^ERRORS:")

    if [[ "$py_line" == PYTHON_OK:* ]]; then
        ok "${py_line#PYTHON_OK:}"
    else
        fail "Python 3.12 не найден"
    fi

    if [[ "$venv_line" == "VENV_OK" ]]; then
        ok "venv"
    else
        fail "venv не найден (~ddl/.venv)"
    fi

    if [[ "$client_line" == "CLIENT_OK" ]]; then
        ok "ddl-client"
    else
        fail "ddl-client не установлен"
    fi

    # Версии
    versions="${ver_line#VERSIONS:}"
    if [[ -n "$versions" && "$versions" != "{}" ]]; then
        echo "  libs: $versions"
    fi

    # Диск
    disk="${disk_line#DISK:}"
    if [[ -n "$disk" ]]; then
        echo "  disk free: $disk"
    fi

    # Итог по узлу
    errors="${err_line#ERRORS:}"
    if [[ -z "${errors// /}" ]]; then
        ((PASS++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
done

echo "============================================"
echo "  Итого: $PASS/$TOTAL OK"
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "  Проблемных узлов: $FAIL_COUNT"
fi
echo "============================================"
