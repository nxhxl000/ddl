# scripts/make_cifar10_splits.py
from __future__ import annotations

# --- добавить корень репозитория в PYTHONPATH ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ------------------------------------------------

import argparse
import json
import numpy as np
from torchvision import datasets

from src.datasets.partition import split_iid, split_dirichlet, save_split


def class_histograms(labels_np: np.ndarray, split: list[list[int]], num_classes: int) -> list[list[int]]:
    """Подсчёт гистограмм классов для каждого клиента."""
    hists: list[list[int]] = []
    for idxs in split:
        if not idxs:
            hists.append([0] * num_classes)
            continue
        cls = labels_np[np.asarray(idxs, dtype=int)]
        hist = np.bincount(cls, minlength=num_classes).astype(int).tolist()
        hists.append(hist)
    return hists


def verify_partition(split: list[list[int]], n: int) -> None:
    """Проверка целостности: покрытие и отсутствие пересечений."""
    if not split:
        raise ValueError("Empty split (no clients).")
    all_idx = np.concatenate([np.asarray(s, dtype=int) for s in split]) if split[0] else np.array([], dtype=int)
    if all_idx.size != n:
        raise AssertionError(f"Total indices {all_idx.size} != {n}")
    if len(np.unique(all_idx)) != n:
        raise AssertionError("Indices are not unique (overlap detected).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="splits")
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--mode", type=str, choices=["iid", "dirichlet"], default="dirichlet")
    ap.add_argument("--alpha", type=float, default=0.3, help="концентрация для Dirichlet (меньше — сильнее non-IID)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_empty", action="store_true", help="разрешить клиентов с 0 образцов в Dirichlet")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Трансформы не нужны — берём только метки
    trainset = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=None)
    # torchvision>=0.10: labels в .targets
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

        # при необходимости запретить пустых клиентов
        if not args.allow_empty:
            empties = [k for k, idx in enumerate(split) if len(idx) == 0]
            if len(empties) > 0:
                raise RuntimeError(
                    f"Found empty clients: {empties}. "
                    f"Use --allow_empty or increase alpha / adjust split_dirichlet to enforce min-per-client."
                )

    # Проверка целостности
    verify_partition(split, n)

    # Класс-статистика
    labels_np = np.asarray(labels, dtype=int)
    hist_overall = np.bincount(labels_np, minlength=num_classes).astype(int).tolist()
    hist_per_client = class_histograms(labels_np, split, num_classes)
    frac_per_client = [
        (np.array(h) / max(1, sum(h))).round(6).tolist() if sum(h) > 0 else [0.0] * num_classes
        for h in hist_per_client
    ]
    dominant_frac = [float(max(h) / max(1, sum(h))) if sum(h) > 0 else 0.0 for h in hist_per_client]

    # Метаданные + статистика (в основном JSON)
    meta = {
        "dataset": "cifar10",
        "mode": args.mode,
        "alpha": (args.alpha if args.mode == "dirichlet" else None),
        "num_clients": args.num_clients,
        "seed": args.seed,
        "num_samples": n,
        "num_classes": num_classes,
        "class_hist_overall": hist_overall,
        "class_hist_per_client": hist_per_client,
        "class_frac_per_client": frac_per_client,
        "dominant_class_fraction_min": min(dominant_frac) if len(dominant_frac) > 0 else 0.0,
        "dominant_class_fraction_max": max(dominant_frac) if len(dominant_frac) > 0 else 0.0,
    }

    # Сохранение разбиения (использует вашу save_split)
    out_path = out_dir / name
    save_split(out_path, split, meta)

    # (опционально) отдельный stats-файл
    stats_path = out_dir / name.replace(".json", "_stats.json")
    stats_payload = {
        "dataset": "cifar10",
        "mode": args.mode,
        "alpha": (args.alpha if args.mode == "dirichlet" else None),
        "num_clients": args.num_clients,
        "seed": args.seed,
        "num_classes": num_classes,
        "class_hist_overall": hist_overall,
        "class_hist_per_client": hist_per_client,
        "class_frac_per_client": frac_per_client,
        "dominant_class_fraction_per_client": dominant_frac,
    }
    stats_path.write_text(json.dumps(stats_payload), encoding="utf-8")

    sizes = [len(s) for s in split]
    print(f"✅ Saved split -> {out_path}")
    print(f"✅ Saved stats -> {stats_path}")
    print(f"Clients: {len(split)} | min/max per-client: {min(sizes)}/{max(sizes)} | total: {sum(sizes)}")
    print(f"Dominant class fraction per client: min={meta['dominant_class_fraction_min']:.3f} "
          f"| max={meta['dominant_class_fraction_max']:.3f}")


if __name__ == "__main__":
    main()
