from __future__ import annotations
from pathlib import Path
import json
import time
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

# --- Константы CIFAR-10 ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

SEED = 42
VAL_SIZE = 5_000  # из 50k train -> 45k train / 5k val

def main():
    torch.manual_seed(SEED)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

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
    train_full = datasets.CIFAR10(str(data_dir), train=True,  download=True, transform=tf_train)
    test_set   = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=tf_test)

    print(f"Num classes: {len(CIFAR10_CLASSES)}")
    print(f"Train (full): {len(train_full)}")
    print(f"Test:         {len(test_set)}")

    # Метки для стратификации
    labels = train_full.targets if hasattr(train_full, "targets") else train_full.labels
    labels = torch.as_tensor(labels)

    # --- Стратифицированный сплит train/val ---
    val_ratio = VAL_SIZE / len(train_full)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    (train_idx, val_idx), = sss.split(torch.zeros(len(labels)), labels)

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()

    # Сохраним индексы на диск (для воспроизводимости и использования в других скриптах)
    split_path = data_dir / "cifar10_splits.json"
    split_payload = {
        "seed": SEED,
        "val_size": VAL_SIZE,
        "timestamp": int(time.time()),
        "classes": list(CIFAR10_CLASSES),
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    print(f"[✓] Saved splits -> {split_path} (train={len(train_idx)}, val={len(val_idx)})")

    # --- Пример сборки DataLoader'ов ---
    num_workers = min(8, os.cpu_count() or 2)
    pin = torch.cuda.is_available()

    train_set = Subset(train_full, train_idx)
    val_set   = Subset(train_full, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pin
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pin
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=pin
    )

    # --- Быстрые sanity-checks ---
    print("\nClass distribution in TRAIN (first 10):")
    counts = [0]*10
    for i in train_idx:
        counts[int(labels[i])] += 1
    for k, (name, cnt) in enumerate(zip(CIFAR10_CLASSES, counts)):
        print(f"[{k:02d}] {name:12s} -> {cnt}")

    xb, yb = next(iter(train_loader))
    assert xb.ndim == 4 and xb.shape[1:] == (3, 32, 32), f"Unexpected shape: {xb.shape}"
    print(f"\nSample batch: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
    print(f"dtype={xb.dtype} range=({float(xb.min()):.3f}, {float(xb.max()):.3f})")
    print("\n✅ CIFAR-10 downloaded, splits saved, loaders ready.")

if __name__ == "__main__":
    main()
