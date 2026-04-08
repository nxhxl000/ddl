"""Profiling round: collect hardware info, data stats, training benchmark.

Клиентская сторона (вызывается из client_app.py):
  collect_hardware_info()       — CPU/RAM через psutil
  collect_data_profile()        — статистика датасета + распределение классов
  run_benchmark()               — мини-обучение для замера скорости узла

Серверная сторона (вызывается из server_app.py):
  run_profiling_round()         — запускает 1 Flower-раунд с флагом profiling-mode
  save_cluster_profile()        — сохраняет cluster_profile.json
  print_profiling_summary()     — таблица профилей в терминал
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


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

    Возвращает значение в [0, 1]: 0 = одинаковые, 1 = максимально разные.
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

    return round(0.5 * _kl(p) + 0.5 * _kl(q), 4)


def _mean_pairwise_js(dists: List[Dict[int, int]], num_classes: int) -> float:
    """Среднее попарное JS-расстояние между всеми парами клиентов."""
    n = len(dists)
    if n < 2:
        return 0.0
    values = [
        _js_divergence(dists[i], dists[j], num_classes)
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return round(sum(values) / len(values), 4)


# ── Клиентские утилиты ────────────────────────────────────────────────────────

def collect_hardware_info() -> Dict[str, float]:
    """Собрать CPU/RAM. Возвращает dict[str, float] для MetricRecord."""
    if not _PSUTIL:
        return {}
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    return {
        "hw_cpu_logical":  float(psutil.cpu_count(logical=True) or 0),
        "hw_cpu_physical": float(psutil.cpu_count(logical=False) or 0),
        "hw_cpu_freq_mhz": float(cpu_freq.current if cpu_freq else 0.0),
        "hw_ram_total_gb": round(mem.total / 1e9, 2),
        "hw_ram_avail_gb": round(mem.available / 1e9, 2),
    }


def collect_data_profile(partition_path: Path | str) -> Dict[str, float]:
    """Статистика датасета: размер, распределение классов, вырожденные классы.

    Ключи вида data_cls_{N} содержат число сэмплов класса N.
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
    # Вырожденный класс: сэмплов < 10% от среднего или < 30
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


def run_benchmark(
    model: nn.Module,
    partition_path: Path | str,
    device: torch.device,
    *,
    max_samples: int = 1000,
    epochs: int = 2,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Замер скорости узла: мини-обучение на max_samples сэмплах.

    Модель изменяется на месте — передавайте копию если нужно сохранить веса.
    """
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    ds = load_from_disk(str(partition_path))
    keys = set(ds.features.keys())
    img_col = next((c for c in ("img", "image", "pixel_values") if c in keys), None)
    label_col = next(c for c in ("label", "labels", "fine_label", "coarse_label") if c in keys)

    n = min(max_samples, len(ds))
    ds = ds.select(range(n))

    def _transform(batch):
        batch[img_col] = [ToTensor()(x) for x in batch[img_col]]
        return batch

    ds = ds.with_transform(_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch_times: list[float] = []
    samples_per_epoch = 0
    final_loss = 0.0

    for _ in range(epochs):
        t0 = time.perf_counter()
        epoch_loss, batches, epoch_samples = 0.0, 0, 0
        for batch in loader:
            x = batch[img_col].to(device)
            y = batch[label_col].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
            epoch_samples += len(y)
        epoch_times.append(time.perf_counter() - t0)
        final_loss = epoch_loss / max(batches, 1)
        samples_per_epoch = epoch_samples

    mean_epoch_sec = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    sps = samples_per_epoch / mean_epoch_sec if mean_epoch_sec > 0 else 0.0

    return {
        "bench_samples":         float(samples_per_epoch),
        "bench_epochs":          float(epochs),
        "bench_mean_epoch_sec":  round(mean_epoch_sec, 3),
        "bench_samples_per_sec": round(sps, 1),
        "bench_time_per_1k_sec": round(1000.0 / sps, 3) if sps > 0 else 0.0,
        "bench_final_loss":      round(final_loss, 4),
    }


# ── Серверная стратегия профилирования ───────────────────────────────────────

class ProfilingStrategy(FedAvg):
    """FedAvg-обёртка для профилировочного раунда.

    Перехватывает aggregate_train и сохраняет полный MetricRecord каждого клиента.
    Агрегированные веса игнорируются — результат профилирования только в метриках.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client_profiles: Dict[int, Dict[str, float]] = {}

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        for rep in replies:
            if not rep.has_content() or "metrics" not in rep.content:
                continue
            m: MetricRecord = rep.content["metrics"]  # type: ignore[assignment]
            cid = int(m.get("partition-id", float(rep.metadata.src_node_id)))
            # Сохраняем все float-поля MetricRecord
            self._client_profiles[cid] = dict(m)
        # Не вызываем super() — агрегация весов не нужна, а FedAvg проверяет
        # одинаковость ключей MetricRecord, что ломается при вырожденных классах
        # (data_cls_N отсутствует у клиентов без этого класса).
        return ArrayRecord({}), MetricRecord({})

    def get_profiles(self) -> Dict[int, Dict[str, float]]:
        return dict(self._client_profiles)


# ── Серверные публичные функции ───────────────────────────────────────────────

def run_profiling_round(
    grid: Any,
    initial_arrays: ArrayRecord,
    *,
    fraction_train: float,
    min_train_nodes: int,
    min_available_nodes: int,
    benchmark_samples: int = 1000,
    benchmark_epochs: int = 2,
) -> Dict[int, Dict[str, float]]:
    """Запустить 1 профилировочный раунд и вернуть профили клиентов.

    Returns:
        {partition_id: {metric_key: float_value, ...}}
    """
    from flwr.app import ConfigRecord

    strategy = ProfilingStrategy(
        fraction_train=fraction_train,
        fraction_evaluate=0.0,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=0,
        min_available_nodes=min_available_nodes,
        weighted_by_key="num-examples",
        arrayrecord_key="arrays",
        configrecord_key="config",
    )

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=1,
        train_config=ConfigRecord({
            "profiling-mode":    1.0,
            "benchmark-samples": float(benchmark_samples),
            "benchmark-epochs":  float(benchmark_epochs),
        }),
        evaluate_fn=lambda rnd, arrays: MetricRecord({}),
    )

    return strategy.get_profiles()


def save_cluster_profile(
    profiles: Dict[int, Dict[str, float]],
    exp_dir: Path,
    *,
    partition_name: str,
    num_classes: int,
) -> Path:
    """Сохранить cluster_profile.json в директорию эксперимента."""
    enriched: Dict[str, Any] = {
        "partition_name": partition_name,
        "num_clients":    len(profiles),
        "clients":        {},
    }

    for cid, metrics in sorted(profiles.items()):
        # Вычленяем per-class distribution из плоских ключей data_cls_N
        class_dist = {
            int(k.split("data_cls_")[1]): int(v)
            for k, v in metrics.items()
            if k.startswith("data_cls_")
        }
        # Всё остальное — скалярные метрики
        scalars = {k: v for k, v in metrics.items() if not k.startswith("data_cls_")}
        scalars["class_distribution"] = class_dist

        # Вырожденные и отсутствующие классы
        if class_dist:
            mean_c = sum(class_dist.values()) / len(class_dist)
            thresh = min(30.0, mean_c * 0.1)
            scalars["degenerate_classes"] = sorted(
                c for c, n in class_dist.items() if n < thresh
            )
            scalars["missing_classes"] = sorted(
                c for c in range(num_classes) if c not in class_dist
            )
            # Per-client: нормализованная энтропия (1=IID, 0=один класс)
            scalars["entropy_norm"] = _entropy_norm(class_dist, num_classes)

        enriched["clients"][str(cid)] = scalars

    # System-level: среднее попарное JS-расстояние между клиентами
    all_dists = [
        enriched["clients"][str(cid)].get("class_distribution", {})
        for cid in sorted(profiles.keys())
    ]
    enriched["mean_pairwise_js"] = _mean_pairwise_js(all_dists, num_classes)

    out_path = exp_dir / "cluster_profile.json"
    out_path.write_text(json.dumps(enriched, indent=2))
    return out_path


def print_profiling_summary(
    profiles: Dict[int, Dict[str, float]],
    *,
    num_classes: int = 0,
) -> None:
    """Печатает таблицу профилей клиентов в терминал.

    num_classes: передаётся для вычисления entropy_norm и mean_pairwise_js.
    """
    width = 78
    print("=" * width)
    print(f"  Профилирование кластера: {len(profiles)} клиентов")
    print("=" * width)
    header = (
        f"  {'ID':>3}  {'CPU':>4}  {'RAM':>6}  {'Сэмплы':>8}"
        f"  {'Неравн.':>8}  {'Вырожд.':>8}  {'Скорость':>10}  {'Loss':>7}"
    )
    if num_classes > 1:
        header += f"  {'Энтропия':>9}"
    print(header)
    print("-" * width)

    all_dists: List[Dict[int, int]] = []
    for cid in sorted(profiles):
        m   = profiles[cid]
        cpu = int(m.get("hw_cpu_logical", 0))
        ram = m.get("hw_ram_total_gb", 0.0)
        n   = int(m.get("data_num_samples", 0))
        imb = m.get("data_imbalance_ratio", 0.0)
        deg = int(m.get("data_n_degenerate", 0))
        sps = m.get("bench_samples_per_sec", 0.0)
        loss = m.get("bench_final_loss", 0.0)

        deg_str = f"{deg} (!)" if deg > 0 else "0"
        line = (
            f"  {cid:>3}  {cpu:>4}  {ram:>5.1f}G"
            f"  {n:>8}  {imb:>7.1f}x  {deg_str:>8}"
            f"  {sps:>8.0f}/s  {loss:>7.4f}"
        )

        if num_classes > 1:
            class_dist = {
                int(k.split("data_cls_")[1]): int(v)
                for k, v in m.items()
                if k.startswith("data_cls_")
            }
            ent = _entropy_norm(class_dist, num_classes)
            line += f"  {ent:>9.4f}"
            all_dists.append(class_dist)

        print(line)

    print("=" * width)

    if num_classes > 1 and len(all_dists) >= 2:
        mpjs = _mean_pairwise_js(all_dists, num_classes)
        print(f"  Межклиентская гетерогенность (mean pairwise JS): {mpjs:.4f}"
              f"  {'(близко к IID)' if mpjs < 0.05 else '(умеренная)' if mpjs < 0.20 else '(высокая)'}")
        print("=" * width)
