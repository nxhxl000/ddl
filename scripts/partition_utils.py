"""
scripts/partition_utils.py — логика разбивки датасета между клиентами FL.

Алгоритм floor+scheme:
  1. Из каждого класса берём min_per_class образцов на клиента и раздаём поровну
     (гарантированный минимум на каждый класс у каждого клиента).
  2. Оставшиеся образцы каждого класса распределяем по схеме:
     - iid:       поровну между всеми клиентами
     - dirichlet: пропорционально выборке из Dir(alpha)

Публичный API:
  extract_server_dataset(dataset, server_size, *, seed, label_col)
      -> tuple[Dataset, Dataset]   # (server_ds, remaining_ds)

  partition_dataset(dataset, num_clients, scheme, *, alpha, min_per_class, seed, label_col)
      -> list[Dataset]

  save_partitions(ds_dict, partitions, out_dir, *, dataset, scheme, alpha,
                  min_per_class, seed, label_col, server_dataset, force)
      -> Path

  partition_dir_name(dataset, scheme, num_clients, seed, *, server_size) -> str

  load_manifest(out_dir) -> dict
"""
from __future__ import annotations

import collections
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_floor(
    indices: np.ndarray,
    num_clients: int,
    min_per_client: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Гарантированно раздаём min_per_client образцов каждому клиенту.

    Если образцов не хватает, actual уменьшается до максимально возможного
    (len(indices) // num_clients), поэтому никогда не падает.

    Returns:
        (floor_splits, remaining) где floor_splits[i] содержит образцы для клиента i,
        remaining — оставшиеся образцы для распределения по схеме.
    """
    rng.shuffle(indices)
    actual = min(min_per_client, len(indices) // num_clients)
    floor_total = actual * num_clients
    floor_idx = indices[:floor_total]
    remaining = indices[floor_total:]
    floor_splits = [floor_idx[i * actual:(i + 1) * actual] for i in range(num_clients)]
    return floor_splits, remaining


def _iid_split(
    indices: np.ndarray,
    num_clients: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Равномерно перемешивает и делит на num_clients частей.
    Остаток (если len не делится нацело) распределяется по одному первым клиентам.
    """
    rng.shuffle(indices)
    return list(np.array_split(indices, num_clients))


def _dirichlet_split(
    indices: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Разбивает пропорционально выборке из Dir(alpha).

    Чем меньше alpha — тем сильнее вырождение (некоторые клиенты получают
    почти все образцы класса, другие почти ничего).
    alpha=1.0 — однородное распределение Dirichlet.
    alpha=100  — близко к IID.
    """
    if len(indices) == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_clients)]

    rng.shuffle(indices)
    proportions = rng.dirichlet([alpha] * num_clients)

    # Целочисленные количества, сумма = len(indices)
    counts = (proportions * len(indices)).astype(int)
    diff = len(indices) - counts.sum()
    # Добавляем остаток клиентам с наибольшей дробной частью
    frac = proportions * len(indices) - counts
    for i in np.argsort(-frac)[:diff]:
        counts[i] += 1

    splits, start = [], 0
    for c in counts:
        splits.append(indices[start:start + c])
        start += c
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_server_dataset(
    dataset: Dataset,
    server_size: int,
    *,
    seed: int = 42,
    label_col: str = "label",
) -> tuple[Dataset, Dataset]:
    """Выделяет сбалансированный серверный датасет из тренировочных данных.

    Стратифицированная выборка: floor(server_size / num_classes) образцов на класс.
    Остаток (dataset минус серверный датасет) возвращается для разбивки между клиентами.

    Args:
        dataset:     Полный тренировочный датасет.
        server_size: Количество образцов для серверного датасета.
                     Должно быть >= num_classes.
        seed:        Seed для воспроизводимости.
        label_col:   Имя столбца с метками.

    Returns:
        (server_dataset, remaining_dataset)
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset[label_col])
    classes = sorted(set(labels.tolist()))
    num_classes = len(classes)

    per_class = server_size // num_classes
    if per_class == 0:
        raise ValueError(
            f"server_size={server_size} слишком мало для {num_classes} классов "
            f"(нужно минимум {num_classes})"
        )

    server_indices: list[int] = []
    remaining_indices: list[int] = []

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        if len(cls_idx) < per_class:
            raise ValueError(
                f"Класс {cls} содержит только {len(cls_idx)} образцов, "
                f"необходимо {per_class} для серверного датасета"
            )
        server_indices.extend(cls_idx[:per_class].tolist())
        remaining_indices.extend(cls_idx[per_class:].tolist())

    actual_size = per_class * num_classes  # может быть меньше server_size на остаток деления
    if actual_size != server_size:
        print(
            f"[partition] server_size={server_size} не делится на {num_classes} классов — "
            f"фактический размер: {actual_size} ({per_class}/класс)"
        )

    server_ds = dataset.select(sorted(server_indices))
    remaining_ds = dataset.select(sorted(remaining_indices))
    return server_ds, remaining_ds


def partition_dataset(
    dataset: Dataset,
    num_clients: int,
    scheme: str = "iid",
    *,
    alpha: float = 0.5,
    min_per_class: int = 0,
    seed: int = 42,
    label_col: str = "label",
) -> list[Dataset]:
    """Разбивает dataset на num_clients партиций с алгоритмом floor+scheme.

    Args:
        dataset:       HuggingFace Dataset (обычно train-сплит).
        num_clients:   количество клиентов.
        scheme:        "iid" или "dirichlet".
        alpha:         параметр Dirichlet (только для scheme="dirichlet").
                       Меньше → сильнее non-IID. Типичные значения: 0.1–1.0.
        min_per_class: минимум образцов каждого класса у каждого клиента.
                       0 — без гарантий (чистый IID/Dirichlet).
        seed:          seed для воспроизводимости.
        label_col:     имя столбца с метками в датасете.

    Returns:
        list[Dataset] длиной num_clients.
    """
    if scheme not in ("iid", "dirichlet"):
        raise ValueError(f"scheme must be 'iid' or 'dirichlet', got '{scheme}'")

    rng = np.random.default_rng(seed)
    labels = np.array(dataset[label_col])
    classes = sorted(set(labels.tolist()))

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]

        # 1. Гарантированный минимум
        floor_splits, remaining = _split_floor(cls_idx, num_clients, min_per_class, rng)

        # 2. Оставшиеся — по схеме
        if scheme == "iid":
            scheme_splits = _iid_split(remaining, num_clients, rng)
        else:
            scheme_splits = _dirichlet_split(remaining, num_clients, alpha, rng)

        # 3. Объединяем
        for i in range(num_clients):
            client_indices[i].extend(floor_splits[i].tolist())
            client_indices[i].extend(scheme_splits[i].tolist())

    return [dataset.select(sorted(idx)) for idx in client_indices]


def save_partitions(
    ds_dict: DatasetDict,
    partitions: list[Dataset],
    out_dir: Path,
    *,
    dataset: str,
    scheme: str,
    alpha: float,
    min_per_class: int,
    seed: int,
    label_col: str = "label",
    server_dataset: "Dataset | None" = None,
    force: bool = False,
) -> Path:
    """Сохраняет партиции и тест-сплит на диск, записывает manifest.json.

    Структура:
        out_dir/
          client_0/   ← HuggingFace Dataset (Arrow format)
          client_1/
          ...
          test/       ← тест-сплит (если есть в ds_dict)
          manifest.json

    Args:
        ds_dict:    DatasetDict (нужен для test-сплита).
        partitions: результат partition_dataset().
        out_dir:    путь для сохранения.
        force:      True — удалить существующую директорию и пересоздать.

    Returns:
        Path к директории партиций.
    """
    out_dir = Path(out_dir)

    if out_dir.exists():
        if force:
            shutil.rmtree(out_dir)
        else:
            print(f"[partition] Already exists: {out_dir}  (use force=True to overwrite)")
            return out_dir

    out_dir.mkdir(parents=True)

    if server_dataset is not None:
        server_dataset.save_to_disk(str(out_dir / "server"))
        print(f"  server:   {len(server_dataset):,} samples (сбалансированный)")

    for i, part in enumerate(partitions):
        part.save_to_disk(str(out_dir / f"client_{i}"))
        print(f"  client_{i}: {len(part):,} samples")

    if "test" in ds_dict:
        ds_dict["test"].save_to_disk(str(out_dir / "test"))
        print(f"  test:     {len(ds_dict['test']):,} samples")

    # Определяем имена классов
    label_feat = partitions[0].features.get(label_col) if partitions else None
    if label_feat and hasattr(label_feat, "names"):
        class_names = list(label_feat.names)
    else:
        all_cls: set[int] = set()
        for p in partitions:
            all_cls.update(p[label_col])
        class_names = [str(c) for c in sorted(all_cls)]

    num_classes = len(class_names)

    # Статистика распределения классов по клиентам
    client_stats = []
    for part in partitions:
        ctr = collections.Counter(part[label_col])
        classes_count = {str(i): int(ctr.get(i, 0)) for i in range(num_classes)}
        client_stats.append({"total": len(part), "classes": classes_count})

    manifest = {
        "dataset":       dataset,
        "scheme":        scheme,
        "alpha":         alpha if scheme == "dirichlet" else None,
        "min_per_class": min_per_class,
        "seed":          seed,
        "num_clients":   len(partitions),
        "num_classes":   num_classes,
        "class_names":   class_names,
        "clients":       client_stats,
        "test_size":     len(ds_dict["test"]) if "test" in ds_dict else None,
        "server_size":   len(server_dataset) if server_dataset is not None else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[partition] Done → {out_dir}")
    return out_dir


def partition_dir_name(
    dataset: str,
    scheme: str,
    num_clients: int,
    seed: int,
    *,
    alpha: float | None = None,
    min_per_class: int = 0,
    server_size: int = 0,
) -> str:
    """Генерирует стандартное имя директории партиций.

    Примеры:
        "cifar100__iid__n6__s42"
        "cifar100__dirichlet__a0.5__m50__n6__s42"
        "cifar100__dirichlet__a0.5__m50__n6__s42__srv10000"
    """
    parts = [dataset, scheme]
    if scheme == "dirichlet":
        parts.append(f"a{alpha}")
        parts.append(f"m{min_per_class}")
    parts.append(f"n{num_clients}")
    parts.append(f"s{seed}")
    if server_size > 0:
        parts.append(f"srv{server_size}")
    return "__".join(parts)


def load_manifest(out_dir: Path) -> dict:
    """Загружает manifest.json из директории партиций."""
    return json.loads((Path(out_dir) / "manifest.json").read_text())
