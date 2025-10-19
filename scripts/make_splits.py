from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from torchvision import datasets
from collections import defaultdict
import time
import yaml


# Функция для IID сплита
def split_iid(n, num_clients, seed=42):
    """Создает IID сплит для клиентов, равномерно распределяя данные."""
    np.random.seed(seed)
    indices = np.random.permutation(n)  # случайная перестановка всех индексов
    split = np.array_split(indices, num_clients)  # Разделение на num_clients частей
    return [list(client_indices) for client_indices in split]


# Функция для Non-IID сплита с использованием распределения Дирихле
def split_dirichlet(labels, num_clients, alpha, num_classes, seed=42):
    """Создает non-IID сплит с использованием распределения Дирихле."""
    # Группируем индексы по классам
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Инициализация генератора случайных чисел
    np.random.seed(seed)

    # Для каждого класса генерируем распределение данных по клиентам с использованием Дирихле
    client_data = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        indices = np.array(class_indices[class_id])  # Преобразуем в NumPy-массив
        num_data_points = len(indices)

        if num_data_points == 0:
            continue  # Если класс пустой, пропускаем

        # Генерируем веса для распределения Дирихле (для каждого клиента)
        dirichlet_weights = np.random.dirichlet([alpha] * num_clients)

        # Определяем количество примеров для каждого клиента (multinomial)
        num_per_client = np.random.multinomial(num_data_points, dirichlet_weights)

        # Перемешиваем индексы для случайного распределения
        np.random.shuffle(indices)

        # Кумулятивные суммы для разбиения
        cumsums = np.cumsum(num_per_client)

        # Разбиваем индексы на части согласно num_per_client
        client_data_points = np.split(indices, cumsums[:-1])

        # Добавляем данные каждому клиенту
        for i in range(num_clients):
            client_data[i].extend(client_data_points[i])

    return client_data


# Функция для сохранения сплита в JSON
def save_split(out_path, split, meta={}):
    """Сохранение разбиения и метаданных в файл"""
    # Преобразуем все элементы в стандартный тип int для сериализации в JSON
    def convert_to_standard_int(value):
        if isinstance(value, np.int64):
            return int(value)
        return value
    
    # Применяем функцию ко всем элементам в сплите
    split = [[convert_to_standard_int(idx) for idx in client] for client in split]

    data = {
        "indices": split,  # Индексы для всех клиентов
        "classes": meta.get("classes", []),  # Список классов
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved split to {out_path}")


# Функция для создания конфигурации Flower
def save_flwr_config(split_path, num_clients, rounds=5, clients_per_round=5, batch_size=64, lr=0.01, out_dir=Path("runconfig")):
    """Создание конфигурации для Flower и сохранение в папку runconfig"""
    config = {
        "rounds": rounds,
        "clients_per_round": clients_per_round,
        "data_dir": "data",  # Папка данных в корне репозитория
        "split_path": str(split_path),
        "batch_size": batch_size,
        "lr": lr,
        "local_epochs": 1  # Можно добавить, если нужно
    }

    # Папка для конфигурации
    out_dir.mkdir(parents=True, exist_ok=True)

    # Извлекаем имя сплита для уникального имени конфига
    split_name = split_path.stem  # Например, "cifar10_iid_K20_seed42"
    config_path = out_dir / f"{split_name}_flwr_config.yaml"
    
    # Сохраняем конфиг в папку runconfig
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Saved Flower config for split '{split_name}' to {config_path}")


# Основной процесс
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")  # Путь к данным (в корне репозитория)
    ap.add_argument("--out_dir", type=str, default="splits")  # Папка для сплитов
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--mode", type=str, choices=["iid", "dirichlet"], default="dirichlet")
    ap.add_argument("--alpha", type=float, default=0.3, help="Концентрация для Dirichlet (меньше — сильнее non-IID)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Путь к данным и папке сплитов в корне репозитория
    data_dir = Path(__file__).resolve().parents[1] / args.data_dir  # Путь к data/ в корне репозитория
    out_dir = Path(__file__).resolve().parents[1] / args.out_dir  # Путь для сплитов в splits/
    out_dir.mkdir(parents=True, exist_ok=True)  # Создаем папку, если она не существует

    # Трансформы не нужны — берём только метки
    trainset = datasets.CIFAR10(str(data_dir), train=True, download=False, transform=None)
    labels = np.asarray(getattr(trainset, "targets", trainset.targets), dtype=int)
    n = len(trainset)
    num_classes = len(getattr(trainset, "classes", list(range(10))))  # 10 для CIFAR-10

    # Генерация сплита
    if args.mode == "iid":
        split = split_iid(n, args.num_clients, seed=args.seed)
        name = f"cifar10_iid_K{args.num_clients}_seed{args.seed}.json"
    else:
        split = split_dirichlet(labels, args.num_clients, args.alpha, num_classes, seed=args.seed)
        name = f"cifar10_dirichlet_a{args.alpha}_K{args.num_clients}_seed{args.seed}.json"

    # Сохранение разбиения в папку
    out_path = out_dir / name
    save_split(out_path, split, meta={"classes": trainset.classes})

    # Создание конфигурации для Flower
    save_flwr_config(Path(out_path), args.num_clients)

    sizes = [len(s) for s in split]
    print(f"✅ Saved split -> {out_path}")
    print(f"Clients: {len(split)} | min/max per-client: {min(sizes)}/{max(sizes)} | total: {sum(sizes)}")


if __name__ == "__main__":
    main()
