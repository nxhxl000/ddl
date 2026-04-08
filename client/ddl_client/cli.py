"""CLI для запуска FL-клиента.

Использование:
    # Production — клиент со своими данными:
    ddl-client start --server 84.201.165.255:9092 --data ~/plant_photos

    # Research — клиент с предразбитой партицией:
    ddl-client start --server 84.201.165.255:9092 \
        --partition-id 0 --num-partitions 6
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _check_supernode() -> str:
    """Найти flower-supernode в PATH."""
    path = shutil.which("flower-supernode")
    if path is None:
        print(
            "Ошибка: flower-supernode не найден.\n"
            "Установите: pip install 'flwr>=1.22.0'",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def _build_node_config(args: argparse.Namespace) -> str:
    """Собрать строку --node-config из аргументов CLI."""
    pairs: list[str] = []

    if args.data is not None:
        # Production: клиент указывает путь к своим данным
        data_path = str(Path(args.data).resolve())
        pairs.append(f"data-dir={data_path}")

    if args.partition_id is not None:
        # Research: клиент — часть контролируемого эксперимента
        pairs.append(f"partition-id={args.partition_id}")
        pairs.append(f"num-partitions={args.num_partitions}")

    return " ".join(pairs)


def cmd_start(args: argparse.Namespace) -> int:
    """Запустить SuperNode."""
    supernode = _check_supernode()

    node_config = _build_node_config(args)

    cmd = [
        supernode,
        "--superlink", args.server,
        "--insecure",
    ]
    if node_config:
        cmd += ["--node-config", node_config]

    print(f"Подключение к серверу {args.server} ...")
    if args.data:
        print(f"Данные: {Path(args.data).resolve()}")
    if args.partition_id is not None:
        print(f"Режим: research (partition-id={args.partition_id})")

    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nОстановлен.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ddl-client",
        description="FL-клиент для классификации болезней растений",
    )
    sub = parser.add_subparsers(dest="command")

    # --- start ---
    p_start = sub.add_parser("start", help="Запустить клиент")
    p_start.add_argument(
        "--server", required=True,
        help="Адрес сервера SuperLink (host:port, например 84.201.165.255:9092)",
    )

    # Production mode
    p_start.add_argument(
        "--data",
        help="Путь к папке с данными клиента (imagefolder: class_name/img.jpg)",
    )

    # Research mode
    p_start.add_argument(
        "--partition-id", type=int, default=None,
        help="ID партиции (research mode)",
    )
    p_start.add_argument(
        "--num-partitions", type=int, default=6,
        help="Общее число партиций (default: 6)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Валидация: нужен хотя бы один режим
    if args.data is None and args.partition_id is None:
        print(
            "Ошибка: укажите --data (production) или --partition-id (research).",
            file=sys.stderr,
        )
        return 1

    # Валидация: --data путь существует
    if args.data is not None:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Ошибка: путь к данным не найден: {data_path}", file=sys.stderr)
            return 1
        if not data_path.is_dir():
            print(f"Ошибка: {data_path} — не директория.", file=sys.stderr)
            return 1

    return cmd_start(args)


if __name__ == "__main__":
    raise SystemExit(main())
