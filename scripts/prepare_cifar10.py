import json
import time
import os
from pathlib import Path

import torch
from torchvision import datasets, transforms

# Константы CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

SEED = 42

def main():
    torch.manual_seed(SEED)

    # Путь для загрузки данных в корневую папку проекта
    data_dir = Path("data")  # Папка для данных будет в корне репозитория
    data_dir.mkdir(parents=True, exist_ok=True)  # Создаём папку data, если не существует

    # Трансформы: без аугментаций (для чистой проверки)
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # --- Загрузка/скачивание ---
    full_dataset = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tf_test)

    # Логируем информацию о данных
    print(f"Num classes: {len(CIFAR10_CLASSES)}")
    print(f"Full dataset size (train): {len(full_dataset)}")  # Тренировочные данные (50,000)
    print(f"Test dataset size: {len(test_set)}")  # Тестовые данные (10,000)

    # Сохраним индексы тренировочного датасета
    train_indices = list(range(len(full_dataset)))

    # Создаём папку для хранения сплитов, если её нет
    splits_dir = Path("splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем индексы тренировочного датасета
    train_split_path = splits_dir / "cifar10_train_split.json"
    train_split_payload = {
        "timestamp": int(time.time()),
        "classes": list(CIFAR10_CLASSES),
        "train_indices": train_indices,
    }
    train_split_path.write_text(json.dumps(train_split_payload), encoding="utf-8")
    print(f"[✓] Saved train split -> {train_split_path} (train={len(train_indices)})")

    # Проверка данных, чтобы убедиться, что индексы сохранены корректно
    check_split(train_split_path, "train")

    print("\n✅ CIFAR-10 downloaded, train split saved.")

def check_split(split_path: Path, dataset_type: str):
    """Проверка наличия данных по сохраненным индексам."""
    try:
        with open(split_path, 'r') as f:
            split_data = json.load(f)
            indices = split_data.get(f"{dataset_type}_indices", None)
            if indices is None:
                print(f"[ERROR] No {dataset_type} indices found in {split_path}.")
                return
            print(f"[✓] {dataset_type.capitalize()} indices loaded successfully.")
            print(f"First 5 indices in {dataset_type} split: {indices[:5]}")
    except FileNotFoundError:
        print(f"[ERROR] Split file {split_path} not found!")
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON in {split_path}.")

if __name__ == "__main__":
    main()
