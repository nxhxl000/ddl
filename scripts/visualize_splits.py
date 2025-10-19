from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


def load_split(path: Path) -> dict:
    """Загружаем json файл сплита и извлекаем необходимые данные."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "indices" in data and "classes" in data:
        return data
    raise ValueError(f"Не найдено необходимых данных в {path}")


def generate_histograms(split, labels, num_classes=10):
    """Генерация гистограмм классов для каждого клиента и общей гистограммы"""
    class_hist_per_client = []
    for client_indices in split:
        hist = [0] * num_classes
        for idx in client_indices:
            label = labels[idx]  # Используем реальную метку из датасета
            hist[label] += 1
        class_hist_per_client.append(hist)
    
    return class_hist_per_client


def ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_clients_size_hist(hist: np.ndarray, out: Path, dpi: int = 140):
    """Гистограмма для количества данных у каждого клиента"""
    sizes = hist.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(sizes)), sizes)
    ax.set_xlabel("Client id")
    ax.set_ylabel("Samples per client")
    ax.set_title("Samples per Client")
    ax.set_ylim(0, max(sizes) + 500)

    plt.tight_layout()
    fig.savefig(out / "clients_sizes.png", dpi=dpi)
    plt.close(fig)


def plot_heatmap_fractions(hists: np.ndarray, class_names: List[str], out: Path, dpi: int = 140):
    sizes = hists.sum(axis=1, keepdims=True).astype(float)
    sizes[sizes == 0] = 1.0
    frac = hists / sizes  # shape: [num_clients, num_classes]
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(frac, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Class fraction")
    plt.xlabel("Class")
    plt.ylabel("Client id")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=40, ha="right")
    plt.title("Class fraction heatmap (clients × classes)")
    plt.tight_layout()
    fig.savefig(out / "clients_class_fraction_heatmap.png", dpi=dpi)
    plt.close(fig)


def plot_stacked_firstN(hists: np.ndarray, class_names: List[str], out: Path, top_n: int = 10, dpi: int = 140):
    n_clients = hists.shape[0]
    if n_clients == 0:
        return
    top_n = max(1, min(top_n, n_clients))
    H = hists[:top_n].astype(float)
    sizes = H.sum(axis=1, keepdims=True)
    sizes[sizes == 0] = 1.0
    frac = H / sizes  # percent stacked
    x = np.arange(top_n)

    fig = plt.figure(figsize=(12, 6))
    bottom = np.zeros(top_n)
    for c in range(frac.shape[1]):
        plt.bar(x, frac[:, c], bottom=bottom, label=class_names[c])
        bottom += frac[:, c]
    plt.xlabel("Client id")
    plt.ylabel("Fraction")
    plt.ylim(0, 1.0)
    plt.title(f"Class mix (stacked) for first {top_n} clients")
    plt.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    fig.savefig(out / f"stacked_class_mix_first_{top_n}.png", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Визуализация метаданных сплитов CIFAR-10 (IID/Dirichlet).")
    ap.add_argument("--split", type=str, required=True,
                    help="Путь к split.json.")
    ap.add_argument("--out", type=str, default="reports/splits",
                    help="Каталог для сохранения графиков.")
    ap.add_argument("--top_n", type=int, default=10,
                    help="Сколько первых клиентов показать в stacked-графике.")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    split_path = Path(args.split)
    out_dir = ensure_out(Path(args.out))

    # Создание папки для конкретного сплита
    split_name = split_path.stem  # Название сплита (например, "cifar10_iid_K20_seed42")
    split_out_dir = out_dir / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    split_data = load_split(split_path)

    # Путь к данным (аналогично make_splits.py)
    data_dir = Path(__file__).resolve().parents[1] / "data"
    trainset = datasets.CIFAR10(str(data_dir), train=True, download=False, transform=None)
    labels = np.asarray(trainset.targets, dtype=int)  # Реальные метки

    # Классы
    class_names = split_data.get("classes", ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

    # Разделение на индексы
    client_indices = split_data["indices"]

    # Генерация гистограмм
    hist_per_client = generate_histograms(client_indices, labels)

    # Графики
    plot_clients_size_hist(np.array(hist_per_client), split_out_dir, dpi=args.dpi)
    plot_heatmap_fractions(np.array(hist_per_client), class_names, split_out_dir, dpi=args.dpi)
    plot_stacked_firstN(np.array(hist_per_client), class_names, split_out_dir, top_n=args.top_n, dpi=args.dpi)

    print(f"[✓] Saved plots to: {split_out_dir.resolve()}")

if __name__ == "__main__":
    main()