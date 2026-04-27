"""Метрики гетерогенности данных + сбор клиентского распределения классов.

Используется на сервере (для расчёта MPJS/Gini) и на клиенте
(`collect_data_profile` шлёт `data_cls_{N}` в r1).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk


# ── Метрики гетерогенности ────────────────────────────────────────────────────

def _entropy_norm(dist: Dict[int, int], num_classes: int) -> float:
    """Нормализованная энтропия Шеннона локального распределения классов.

    Учитывает все num_classes классов (включая отсутствующие с вероятностью 0).
    Возвращает значение в [0, 1]: 1 = равномерное (IID), 0 = один класс.
    """
    n = sum(dist.values())
    if n == 0 or num_classes <= 1:
        return 0.0
    entropy = -sum(
        (dist.get(c, 0) / n) * math.log(dist.get(c, 0) / n)
        for c in range(num_classes)
        if dist.get(c, 0) > 0
    )
    return round(entropy / math.log(num_classes), 4)


def _js_divergence(dist_a: Dict[int, int], dist_b: Dict[int, int], num_classes: int) -> float:
    """Jensen-Shannon дивергенция между двумя распределениями классов (счётчики).

    Нормирована на ln 2, поэтому возвращает значение в [0, 1]:
    0 = одинаковые, 1 = непересекающиеся носители.
    """
    n_a = sum(dist_a.values())
    n_b = sum(dist_b.values())
    if n_a == 0 or n_b == 0:
        return 1.0
    p = {c: dist_a.get(c, 0) / n_a for c in range(num_classes)}
    q = {c: dist_b.get(c, 0) / n_b for c in range(num_classes)}
    m = {c: (p[c] + q[c]) / 2 for c in range(num_classes)}

    def _kl(a: Dict[int, float]) -> float:
        return sum(a[c] * math.log(a[c] / m[c]) for c in range(num_classes) if a[c] > 0)

    js_nats = 0.5 * _kl(p) + 0.5 * _kl(q)
    return round(js_nats / math.log(2), 4)


def _mean_pairwise_js(dists: List[Dict[int, int]], num_classes: int) -> float:
    """Среднее попарное JS-расстояние между всеми парами клиентов (формула 23)."""
    n = len(dists)
    if n < 2:
        return 0.0
    values = [
        _js_divergence(dists[i], dists[j], num_classes)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return round(sum(values) / len(values), 4)


def _gini_sizes(dists: List[Dict[int, int]]) -> float:
    """Коэффициент Джини для объёмов клиентских датасетов — quantity skew (формула 24).

    sizes[i] = sum(dists[i]) — общее число сэмплов у клиента i.
    0 = все клиенты имеют одинаковый объём; 1 = один клиент владеет всеми данными.
    Ортогонален MPJS (label skew).
    """
    sizes = sorted(sum(d.values()) for d in dists)
    n = len(sizes)
    total = sum(sizes)
    if n < 2 or total == 0:
        return 0.0
    cum = sum(i * s for i, s in enumerate(sizes, 1))
    gini = (2 * cum) / (n * total) - (n + 1) / n
    return round(gini, 4)


def _class_monopoly_index(dists: List[Dict[int, int]], num_classes: int) -> float:
    """Class Monopoly Index (CMI) — структурная монополия классов среди клиентов.

    Для каждого класса k: share_k[i] = count_i[k] / total_k.
    Метрика = 1 − mean_k H(share_k) / log(N_clients).
    0 = каждый класс равномерно у всех клиентов (IID);
    1 = каждый класс у одного клиента (disjoint).

    Дополняет MPJS: разделяет структурную монополию (высокий CMI) и статистический
    перекос с размазанным владением (низкий CMI при высоком MPJS).
    """
    n_clients = len(dists)
    if n_clients < 2 or num_classes <= 0:
        return 0.0
    log_n = math.log(n_clients)
    values = []
    for c in range(num_classes):
        counts = [d.get(c, 0) for d in dists]
        total = sum(counts)
        if total == 0:
            continue
        entropy = -sum((x / total) * math.log(x / total) for x in counts if x > 0)
        values.append(entropy / log_n)
    if not values:
        return 0.0
    return round(1.0 - sum(values) / len(values), 4)


# ── Клиентский сбор распределения классов ────────────────────────────────────

def collect_data_profile(partition_path: Path | str) -> Dict[str, float]:
    """Статистика клиентского датасета: размер + распределение классов.

    Возвращает dict с ключами:
      data_num_samples, data_n_classes, data_imbalance_ratio,
      data_max_class_count, data_min_class_count, data_mean_class_count,
      data_n_degenerate, data_cls_{N} (для каждого присутствующего класса).

    Сервер на round 1 читает `data_cls_{N}` из MetricRecord для расчёта MPJS/Gini.
    """
    ds = load_from_disk(str(partition_path))
    keys = set(ds.features.keys())
    label_col = next(c for c in ("label", "labels", "fine_label", "coarse_label") if c in keys)
    labels = ds[label_col]

    num_samples = len(labels)
    class_counts: Dict[int, int] = {}
    for lbl in labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    n_classes = len(class_counts)
    counts = list(class_counts.values())
    max_c = max(counts) if counts else 0
    min_c = min(counts) if counts else 0
    mean_c = num_samples / n_classes if n_classes > 0 else 0.0
    imbalance = round(max_c / min_c, 3) if min_c > 0 else 0.0
    degen_thresh = min(30.0, mean_c * 0.1)
    n_degen = sum(1 for n in counts if n < degen_thresh)

    profile: Dict[str, float] = {
        "data_num_samples":      float(num_samples),
        "data_n_classes":        float(n_classes),
        "data_imbalance_ratio":  imbalance,
        "data_max_class_count":  float(max_c),
        "data_min_class_count":  float(min_c),
        "data_mean_class_count": round(mean_c, 1),
        "data_n_degenerate":     float(n_degen),
    }
    for cls_id, count in class_counts.items():
        profile[f"data_cls_{cls_id}"] = float(count)
    return profile
