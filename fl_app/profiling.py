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


def _gini_sizes(dists: List[Dict[int, int]]) -> float:
    """Коэффициент Джини для объёмов клиентских датасетов (quantity skew).

    sizes[i] = sum(dists[i]) — общее число сэмплов у клиента i.
    Возвращает значение в [0, 1]:
      0 = все клиенты имеют одинаковый объём (IID по объёму);
      1 = один клиент владеет всеми данными.

    Ортогональна MPJS (label skew) и COE (структурная монополия классов):
    чистый quantity skew даёт Gini > 0 при MPJS ≈ 0 и COE ≈ 1.
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

    Для каждого класса k: share_k[i] = count_i[k] / total_k — доля класса k у клиента i.
    Метрика = 1 − mean_k H(share_k) / log(N_clients).

    Возвращает значение в [0, 1] (единообразно с MPJS и Gini):
      0 = каждый класс равномерно представлен у всех клиентов (IID);
      1 = каждый класс принадлежит ровно одному клиенту (disjoint).

    Дополняет MPJS: разделяет структурную монополию (высокий CMI) и статистический
    перекос с размазанным владением (низкий CMI при высоком MPJS).
    Конструктивно — 1 минус дуал `_entropy_norm` по транспонированной матрице client×class.
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
        entropy = -sum(
            (x / total) * math.log(x / total) for x in counts if x > 0
        )
        values.append(entropy / log_n)
    if not values:
        return 0.0
    return round(1.0 - sum(values) / len(values), 4)


# ── Клиентские утилиты ────────────────────────────────────────────────────────

def _cpu_freq_from_proc() -> float:
    """Fallback: средняя частота CPU из /proc/cpuinfo (Linux). 0.0 если не найдено."""
    try:
        freqs: List[float] = []
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    freqs.append(float(line.split(":")[1].strip()))
        return sum(freqs) / len(freqs) if freqs else 0.0
    except Exception:
        return 0.0


def _cpu_count_from_proc(logical: bool) -> int:
    """Fallback: число CPU из /proc/cpuinfo. logical=True → 'processor', False → уникальные core id."""
    try:
        if logical:
            with open("/proc/cpuinfo") as f:
                return sum(1 for line in f if line.startswith("processor"))
        # Физические: считаем уникальные пары (physical id, core id)
        cores = set()
        phys_id = core_id = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys_id = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":")[1].strip()
                elif line.strip() == "" and phys_id is not None and core_id is not None:
                    cores.add((phys_id, core_id))
                    phys_id = core_id = None
        return len(cores) if cores else 0
    except Exception:
        return 0


def collect_hardware_info() -> Dict[str, float]:
    """Собрать CPU/RAM. Возвращает dict[str, float] для MetricRecord.

    Fallback: при отсутствии psutil или возврате None из psutil.cpu_freq()
    читаем /proc/cpuinfo и /proc/meminfo напрямую.
    """
    cpu_logical  = 0
    cpu_physical = 0
    cpu_freq_mhz = 0.0
    ram_total_gb = 0.0
    ram_avail_gb = 0.0

    if _PSUTIL:
        try:
            cpu_logical  = int(psutil.cpu_count(logical=True)  or 0)
            cpu_physical = int(psutil.cpu_count(logical=False) or 0)
            cf = psutil.cpu_freq()
            if cf is not None and cf.current:
                cpu_freq_mhz = float(cf.current)
            mem = psutil.virtual_memory()
            ram_total_gb = round(mem.total     / 1e9, 2)
            ram_avail_gb = round(mem.available / 1e9, 2)
        except Exception:
            pass

    if cpu_logical == 0:
        cpu_logical = _cpu_count_from_proc(logical=True)
    if cpu_physical == 0:
        cpu_physical = _cpu_count_from_proc(logical=False) or cpu_logical
    if cpu_freq_mhz == 0.0:
        cpu_freq_mhz = _cpu_freq_from_proc()
    if ram_total_gb == 0.0:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_total_gb = round(int(line.split()[1]) * 1024 / 1e9, 2)
                    elif line.startswith("MemAvailable:"):
                        ram_avail_gb = round(int(line.split()[1]) * 1024 / 1e9, 2)
        except Exception:
            pass

    return {
        "hw_cpu_logical":  float(cpu_logical),
        "hw_cpu_physical": float(cpu_physical),
        "hw_cpu_freq_mhz": float(cpu_freq_mhz),
        "hw_ram_total_gb": float(ram_total_gb),
        "hw_ram_avail_gb": float(ram_avail_gb),
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
    batch_times: list[float] = []
    samples_per_epoch = 0

    for _ in range(epochs):
        t0 = time.perf_counter()
        epoch_samples = 0
        for batch in loader:
            tb = time.perf_counter()
            x = batch[img_col].to(device)
            y = batch[label_col].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch_times.append(time.perf_counter() - tb)
            epoch_samples += len(y)
        epoch_times.append(time.perf_counter() - t0)
        samples_per_epoch = epoch_samples

    mean_epoch_sec = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    sps = samples_per_epoch / mean_epoch_sec if mean_epoch_sec > 0 else 0.0

    if batch_times:
        mean_batch_sec = sum(batch_times) / len(batch_times)
        var_batch = sum((t - mean_batch_sec) ** 2 for t in batch_times) / len(batch_times)
        cv_batch = (math.sqrt(var_batch) / mean_batch_sec) if mean_batch_sec > 0 else 0.0
    else:
        cv_batch = 0.0

    return {
        "bench_samples_per_sec": round(sps, 1),
        "bench_cv_batch":        round(cv_batch, 4),
        "bench_n_batches":       float(len(batch_times)),
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
    enriched["mean_pairwise_js"]    = _mean_pairwise_js(all_dists, num_classes)
    enriched["class_monopoly_index"] = _class_monopoly_index(all_dists, num_classes)
    enriched["gini_sizes"]           = _gini_sizes(all_dists)

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
        f"  {'Скорость':>10}  {'CV':>6}"
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
        sps = m.get("bench_samples_per_sec", 0.0)
        cv  = m.get("bench_cv_batch", 0.0)

        line = (
            f"  {cid:>3}  {cpu:>4}  {ram:>5.1f}G"
            f"  {n:>8}  {sps:>8.0f}/s  {cv:>5.2%}"
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
        cmi  = _class_monopoly_index(all_dists, num_classes)
        gini = _gini_sizes(all_dists)
        print(f"  Mean pairwise JS (MPJS, label skew):              {mpjs:.4f}"
              f"  {'(близко к IID)' if mpjs < 0.05 else '(умеренная)' if mpjs < 0.20 else '(высокая)'}")
        print(f"  Class Monopoly Index (CMI, structural):           {cmi:.4f}"
              f"  {'(размазано по клиентам)' if cmi < 0.20 else '(умеренная монополия)' if cmi < 0.70 else '(структурная монополия)'}")
        print(f"  Gini of client sizes (quantity skew):             {gini:.4f}"
              f"  {'(равные объёмы)' if gini < 0.05 else '(умеренное неравенство)' if gini < 0.25 else '(сильное неравенство)'}")
        print("=" * width)
