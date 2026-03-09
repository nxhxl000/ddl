"""Adaptive per-client training parameters based on cluster profile.

Algorithm (straggler mitigation):
  T_target = training time of the fastest client at base_epochs.

  For each client i:
    1. total_budget  = T_target / bench_time_per_1k * 1000   (samples client can process)
    2. ideal_epochs  = total_budget / data_num_samples        (fractional)
    3. epochs        = ceil(ideal_epochs), capped at max_epochs
    4. sample_budget = total_budget / epochs                  (samples per epoch)
       - If sample_budget >= data_num_samples: no chunk needed (use -1)
       - Else: chunk dataset to sample_budget samples per epoch

  Using ceil() instead of floor() eliminates idle time from fractional epoch truncation.
  All clients reach T_target ± measurement noise.

  Data subsampling uses deterministic shuffle (seed = server_round * 100 + partition_id)
  to avoid class imbalance bias from sequential HuggingFace storage.
"""
from __future__ import annotations

import math
from typing import Dict

_NO_ADAPT: Dict = {"local_epochs": -1, "sample_budget": -1, "est_time_sec": -1.0}


def compute_adaptive_params(
    profiles: Dict[int, Dict[str, float]],
    base_epochs: int,
    *,
    max_epochs: int | None = None,
    min_samples: int = 64,
) -> Dict[int, Dict]:
    """Compute per-client local_epochs and sample_budget from cluster profile.

    Args:
        profiles:    {partition_id: metrics_dict} from run_profiling_round()
        base_epochs: default local epochs from pyproject.toml
        max_epochs:  cap on epochs per client (default: base_epochs * 2)
        min_samples: minimum sample_budget when chunking data

    Returns:
        {partition_id: {"local_epochs": int, "sample_budget": int, "est_time_sec": float}}
        sample_budget == -1 means "use all data" (no chunking).
    """
    if max_epochs is None:
        max_epochs = base_epochs * 2

    # Estimate time for 1 full epoch on actual client data
    t_per_epoch: Dict[int, float] = {}
    for cid, m in profiles.items():
        t1k = m.get("bench_time_per_1k_sec", 0.0)
        n   = m.get("data_num_samples", 0.0)
        if t1k > 0 and n > 0:
            t_per_epoch[cid] = t1k * n / 1000.0

    if not t_per_epoch:
        return {cid: dict(_NO_ADAPT) for cid in profiles}

    # T_target: fastest client's estimated training time at base_epochs
    T_target = min(t * base_epochs for t in t_per_epoch.values())

    result: Dict[int, Dict] = {}
    for cid, m in profiles.items():
        if cid not in t_per_epoch:
            result[cid] = dict(_NO_ADAPT)
            continue

        t1k = m["bench_time_per_1k_sec"]
        n   = int(m.get("data_num_samples", 0))

        # Total samples this client can process in T_target time
        total_budget = T_target / t1k * 1000.0

        # How many epochs fit: ceil to eliminate idle time from truncation.
        # Subtract small epsilon before ceil to avoid floating-point artifacts
        # (e.g. T_target / t1k * 1000 / n = 3.0000000001 → ceil = 4 instead of 3).
        ideal_epochs = total_budget / n
        epochs = min(max_epochs, max(1, math.ceil(ideal_epochs - 1e-9)))

        # Samples per epoch to stay within T_target
        budget_per_epoch = total_budget / epochs
        if budget_per_epoch >= n:
            budget = -1                              # full dataset, no chunk
            est = t1k * n / 1000.0 * epochs
        else:
            budget = max(min_samples, int(budget_per_epoch))
            est    = t1k * budget / 1000.0 * epochs

        result[cid] = {
            "local_epochs":  epochs,
            "sample_budget": budget,
            "est_time_sec":  round(est, 2),
        }

    return result


def to_train_config_dict(params: Dict[int, Dict]) -> Dict[str, float]:
    """Flatten adaptive params into ConfigRecord-compatible flat dict.

    Keys: c{pid}_epochs, c{pid}_samples  (float values for ConfigRecord).
    """
    flat: Dict[str, float] = {}
    for pid, p in params.items():
        flat[f"c{pid}_epochs"]  = float(p["local_epochs"])
        flat[f"c{pid}_samples"] = float(p["sample_budget"])
    return flat


def print_adaptive_summary(
    params: Dict[int, Dict],
    profiles: Dict[int, Dict[str, float]],
    base_epochs: int,
) -> None:
    """Print per-client adaptive training schedule to stdout."""
    # Recompute T_target for display
    t_per_epoch = {}
    for cid, m in profiles.items():
        t1k = m.get("bench_time_per_1k_sec", 0.0)
        n   = m.get("data_num_samples", 0.0)
        if t1k > 0 and n > 0:
            t_per_epoch[cid] = t1k * n / 1000.0
    T_target = min(t * base_epochs for t in t_per_epoch.values()) if t_per_epoch else 0.0

    width = 84
    print("=" * width)
    print(f"  Адаптивное расписание  (base_epochs={base_epochs}, T_target={T_target:.1f}s)")
    print("=" * width)
    print(f"  {'ID':>3}  {'Скорость':>10}  {'Сэмплов':>8}  {'Идеал':>6}  {'Эпох':>5}  {'Чанк':>8}  {'Оценка':>8}")
    print("-" * width)

    for cid in sorted(params):
        p   = params[cid]
        m   = profiles.get(cid, {})
        sps    = m.get("bench_samples_per_sec", 0.0)
        n      = int(m.get("data_num_samples", 0))
        t1k    = m.get("bench_time_per_1k_sec", 0.0)
        epochs = p["local_epochs"]
        budget = p["sample_budget"]
        est    = p["est_time_sec"]

        ideal = (T_target / t1k * 1000.0 / n) if (t1k > 0 and n > 0) else 0.0
        budget_str = f"{budget}" if budget != -1 else "все"
        est_str    = f"{est:.1f}s" if est != -1.0 else "?"
        print(
            f"  {cid:>3}  {sps:>8.0f}/s  {n:>8}  {ideal:>5.2f}x"
            f"  {epochs:>5}  {budget_str:>8}  {est_str:>8}"
        )

    print("=" * width)
