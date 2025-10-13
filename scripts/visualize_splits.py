from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def load_stats(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # поддержка обоих форматов: отдельный *_stats.json или meta внутри split.json
    if "class_hist_per_client" in data:
        return data
    if "meta" in data and isinstance(data["meta"], dict):
        meta = data["meta"]
        required = ["class_hist_per_client", "class_hist_overall"]
        if all(k in meta for k in required):
            return meta
    raise ValueError(
        f"Не найден блок статистики в {path} "
        "(ожидались ключи 'class_hist_per_client'/'class_hist_overall' либо meta с этими полями)."
    )


def ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_overall_hist(overall: List[int], class_names: List[str], out: Path, dpi: int = 140):
    x = np.arange(len(overall))
    fig = plt.figure(figsize=(10, 4))
    plt.bar(x, overall)
    plt.xticks(x, class_names, rotation=40, ha="right")
    plt.ylabel("Samples")
    plt.title("Overall class histogram")
    plt.tight_layout()
    fig.savefig(out / "overall_class_hist.png", dpi=dpi)
    plt.close(fig)


def plot_clients_size_hist(hists: np.ndarray, out: Path, dpi: int = 140):
    sizes = hists.sum(axis=1)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(sizes)), sizes)
    plt.xlabel("Client id")
    plt.ylabel("Samples per client")
    plt.title("Client sizes")
    plt.tight_layout()
    fig.savefig(out / "clients_sizes.png", dpi=dpi)
    plt.close(fig)


def plot_dominant_fraction(hists: np.ndarray, out: Path, dpi: int = 140):
    sizes = hists.sum(axis=1).clip(min=1)
    dom = (hists.max(axis=1) / sizes)
    order = np.argsort(dom)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(dom)), dom[order])
    plt.xlabel("Client (sorted by dominance)")
    plt.ylabel("Dominant class fraction")
    plt.ylim(0, 1.0)
    plt.title("Dominant class fraction per client")
    plt.tight_layout()
    fig.savefig(out / "dominant_fraction_per_client.png", dpi=dpi)
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
    ap.add_argument("--stats", type=str, required=True,
                    help="Путь к *_stats.json или к split.json (если в meta есть статистика).")
    ap.add_argument("--out", type=str, default="reports/splits",
                    help="Каталог для сохранения графиков.")
    ap.add_argument("--top_n", type=int, default=10,
                    help="Сколько первых клиентов показать в stacked-графике.")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    stats_path = Path(args.stats)
    out_dir = ensure_out(Path(args.out))

    stats = load_stats(stats_path)

    # Классы
    if "classes" in stats and isinstance(stats["classes"], list):
        class_names = stats["classes"]
    else:
        # CIFAR-10 по умолчанию
        class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    # Гистограммы
    per_client = np.asarray(stats["class_hist_per_client"], dtype=int)  # [K, C]
    overall = np.asarray(stats["class_hist_overall"], dtype=int)        # [C]

    # Графики
    plot_overall_hist(overall.tolist(), class_names, out_dir, dpi=args.dpi)
    plot_clients_size_hist(per_client, out_dir, dpi=args.dpi)
    plot_dominant_fraction(per_client, out_dir, dpi=args.dpi)
    plot_heatmap_fractions(per_client, class_names, out_dir, dpi=args.dpi)
    plot_stacked_firstN(per_client, class_names, out_dir, top_n=args.top_n, dpi=args.dpi)

    # Консольный summary
    sizes = per_client.sum(axis=1)
    dom = per_client.max(axis=1) / np.clip(sizes, 1, None)
    print("\nSummary:")
    print(f"- clients: {per_client.shape[0]} | classes: {per_client.shape[1]}")
    print(f"- samples per client: min={sizes.min()}  p50={np.median(sizes)}  max={sizes.max()}")
    print(f"- dominant frac per client: min={dom.min():.3f}  p50={np.median(dom):.3f}  max={dom.max():.3f}")
    print(f"[✓] Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
