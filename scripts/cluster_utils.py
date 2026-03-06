"""
scripts/cluster_utils.py — утилиты для работы с кластером FL.

Используется в cluster.ipynb и других Python-скриптах.
Bash-скрипты в deploy/ используют аналогичную логику через common.sh.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


# ==============================================================================
# Загрузка конфига
# ==============================================================================

def load_nodes(conf_path: str = "deploy/nodes.conf") -> dict[str, Any]:
    """Читает nodes.conf и возвращает структуру с узлами кластера."""
    raw: dict[str, str] = {}
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            raw[key.strip()] = val.strip()

    clients = []
    i = 1
    while True:
        ext_ip = raw.get(f"CLIENT_{i}")
        if not ext_ip:
            break
        clients.append({
            "name":         raw.get(f"CLIENT_{i}_NAME", f"client-{i}"),
            "ip":           raw.get(f"CLIENT_{i}_INT", ext_ip),
            "ext_ip":       ext_ip,
            "role":         "client",
            "partition_id": i - 1,
            "cores":        int(raw.get(f"CLIENT_{i}_CORES", 1)),
        })
        i += 1

    server = {
        "name": "fl-server", "ip": raw["SERVER_EXT"],
        "ext_ip": raw["SERVER_EXT"], "role": "server", "partition_id": None,
    }

    ssh_key = Path.home() / ".ssh" / "admin-fl"

    return {
        "server":      server,
        "clients":     clients,
        "all_nodes":   [server] + clients,
        "ssh_user":    raw.get("SSH_USER", "gleb"),
        "ssh_key":     str(ssh_key),
        "ssh_key_win": raw.get("SSH_KEY_WIN", "/mnt/c/Users/listr/.ssh/admin-fl"),
        "server_ext":  raw["SERVER_EXT"],
        "server_int":  raw.get("SERVER_INT", raw["SERVER_EXT"]),
        "remote_dir":  raw.get("REMOTE_DIR", "/home/gleb/ddl"),
    }


def setup_ssh_key(cfg: dict[str, Any]) -> None:
    """Копирует SSH-ключ из Windows в WSL если нужно. Вызывать явно перед SSH-командами."""
    ssh_key = Path(cfg["ssh_key"])
    ssh_key_win = Path(cfg["ssh_key_win"])
    if not ssh_key_win.exists():
        return
    if ssh_key.exists() and ssh_key.read_bytes() == ssh_key_win.read_bytes():
        return
    ssh_key.parent.mkdir(exist_ok=True)
    ssh_key.write_bytes(ssh_key_win.read_bytes())
    ssh_key.chmod(0o600)
    print(f"SSH key copied to {ssh_key}")


# ==============================================================================
# SSH
# ==============================================================================

_BASE_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
]


def ssh_server(cfg: dict, cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Выполняет команду на сервере напрямую."""
    return subprocess.run(
        ["ssh", "-i", cfg["ssh_key"], *_BASE_SSH_OPTS,
         f"{cfg['ssh_user']}@{cfg['server_ext']}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def ssh_client(cfg: dict, int_ip: str, cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Выполняет команду на клиенте через сервер как jump host.

    Используем явный ProxyCommand вместо -J, чтобы ключ -i передавался
    и для jump-соединения (OpenSSH не пробрасывает -i в implicit ProxyCommand).
    """
    proxy = (f"ssh -i {cfg['ssh_key']} "
             + " ".join(_BASE_SSH_OPTS)
             + f" -W %h:%p {cfg['ssh_user']}@{cfg['server_ext']}")
    return subprocess.run(
        ["ssh", "-i", cfg["ssh_key"], *_BASE_SSH_OPTS,
         "-o", f"ProxyCommand={proxy}",
         f"{cfg['ssh_user']}@{int_ip}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def ssh_node(cfg: dict, node: dict, cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Выполняет команду на узле (сервер или клиент), выбирает способ подключения автоматически."""
    if node["role"] == "server":
        return ssh_server(cfg, cmd, timeout=timeout)
    return ssh_client(cfg, node["ip"], cmd, timeout=timeout)


# ==============================================================================
# Rsync
# ==============================================================================

_RSYNC_EXCLUDES = [
    "--exclude", ".venv",
    "--exclude", "__pycache__",
    "--exclude", "*.pyc",
    "--exclude", ".git",
    "--exclude", "data/",
    "--exclude", "runs/",
    "--exclude", "local_exp/",
    "--exclude", "models/",
]


def rsync_to_server(cfg: dict, src: str, dst: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """rsync локальной папки src на сервер в dst."""
    ssh_cmd = f"ssh -i {cfg['ssh_key']} " + " ".join(_BASE_SSH_OPTS)
    remote = f"{cfg['ssh_user']}@{cfg['server_ext']}:{dst}"
    return subprocess.run(
        ["rsync", "-az", "-e", ssh_cmd, src, remote],
        timeout=timeout,
    )


def rsync_from_server(cfg: dict, src: str, dst: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """rsync папки src с сервера на локальную машину в dst."""
    ssh_cmd = f"ssh -i {cfg['ssh_key']} " + " ".join(_BASE_SSH_OPTS)
    remote = f"{cfg['ssh_user']}@{cfg['server_ext']}:{src}"
    return subprocess.run(
        ["rsync", "-az", "--info=progress2", "-e", ssh_cmd, remote, dst],
        timeout=timeout,
    )


def rsync_to_client(cfg: dict, int_ip: str, src: str, dst: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """rsync локальной папки src на клиент (через jump host) в dst."""
    proxy = (f"ssh -i {cfg['ssh_key']} "
             + " ".join(_BASE_SSH_OPTS)
             + f" -W %h:%p {cfg['ssh_user']}@{cfg['server_ext']}")
    ssh_cmd = (f"ssh -i {cfg['ssh_key']} "
               + " ".join(_BASE_SSH_OPTS)
               + f" -o 'ProxyCommand={proxy}'")
    remote = f"{cfg['ssh_user']}@{int_ip}:{dst}"
    return subprocess.run(
        ["rsync", "-az", "-e", ssh_cmd, src, remote],
        timeout=timeout,
    )
