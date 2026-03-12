"""Adaptive per-client training parameters based on cluster profile.

Algorithm (straggler mitigation):
  T_target = training time of the fastest client at base_epochs (full data).
  tolerance controls the SPREAD between clients:
    T_min = T_target * (1 - tolerance)   — clients should not finish earlier
    T_max = T_target * (1 + tolerance)   — clients should not finish later

  maximize-epochs  (default):
    epochs = base_epochs (fixed for all clients).
    If client fits within [T_min, T_max] on full data → no chunk.
    Otherwise → chunk so that base_epochs epochs take exactly T_target.
    Note: since T_target = min time, no client on full data can be < T_min,
    so only the upper bound matters here.

  maximize-chunk:
    Full dataset preferred (budget = -1), epochs adjusted.
    1. full_epochs = min(base_epochs, floor(T_max / t_epoch))
    2. If t_epoch * full_epochs in [T_min, T_max] → use full data.
    3. If below T_min (client too fast with full_epochs) →
       add 1 more epoch and chunk to T_target.
    4. If above T_max (even 1 epoch too slow) →
       1 epoch, chunk to T_max.

  Data subsampling uses deterministic shuffle (seed = server_round * 100 + partition_id)
  to avoid class imbalance bias from sequential HuggingFace storage.
"""
from __future__ import annotations

import math
from typing import Any, Dict

_NO_ADAPT: Dict = {"local_epochs": -1, "sample_budget": -1, "est_time_sec": -1.0}


def _t_per_epoch(profiles: Dict[int, Dict]) -> Dict[int, float]:
    """Estimate training time per full epoch for each client."""
    return {
        cid: m["bench_time_per_1k_sec"] * m["data_num_samples"] / 1000.0
        for cid, m in profiles.items()
        if m.get("bench_time_per_1k_sec", 0.0) > 0 and m.get("data_num_samples", 0.0) > 0
    }


def compute_adaptive_params(
    profiles: Dict[int, Dict[str, float]],
    base_epochs: int,
    *,
    mode: str = "maximize-epochs",
    tolerance: float = 0.10,
    min_samples: int = 64,
) -> Dict[int, Dict]:
    """Compute per-client local_epochs and sample_budget from cluster profile.

    Args:
        profiles:    {partition_id: metrics_dict} from run_profiling_round()
        base_epochs: default local epochs from pyproject.toml
        mode:        "maximize-epochs" — fixed epochs=base_epochs, vary chunk size;
                     "maximize-chunk"  — full dataset preferred, vary epochs
        tolerance:   allowed relative spread between fastest and slowest client.
                     T_min = T_target*(1-tol), T_max = T_target*(1+tol).
        min_samples: minimum sample_budget when chunking data

    Returns:
        {partition_id: {"local_epochs": int, "sample_budget": int, "est_time_sec": float}}
        sample_budget == -1 means "use all data" (no chunking).
    """
    if mode not in ("maximize-epochs", "maximize-chunk"):
        raise ValueError(f"Unknown adaptive mode: {mode!r}. Use 'maximize-epochs' or 'maximize-chunk'.")

    tpe = _t_per_epoch(profiles)
    if not tpe:
        return {cid: dict(_NO_ADAPT) for cid in profiles}

    # T_target: fastest client's full training time at base_epochs
    T_target = min(t * base_epochs for t in tpe.values())
    T_min    = T_target * (1.0 - tolerance)
    T_max    = T_target * (1.0 + tolerance)

    result: Dict[int, Dict] = {}
    for cid, m in profiles.items():
        if cid not in tpe:
            result[cid] = dict(_NO_ADAPT)
            continue

        t1k     = m["bench_time_per_1k_sec"]
        n       = int(m.get("data_num_samples", 0))
        t_epoch = tpe[cid]

        if mode == "maximize-epochs":
            # All clients get base_epochs; only chunk size varies.
            # Since T_target = min time, no client can be below T_min on full data.
            epochs = base_epochs
            t_full = t_epoch * epochs
            if t_full <= T_max:
                budget = -1
                est    = t_full
            else:
                # Chunk so that base_epochs epochs fit in exactly T_target
                budget = max(min_samples, int(T_target / epochs / t1k * 1000))
                est    = t1k * budget / 1000.0 * epochs

        else:  # maximize-chunk
            # Prefer full data; vary epochs to keep within [T_min, T_max].
            full_epochs   = min(base_epochs, max(1, math.floor(T_max / t_epoch)))
            t_full_plan   = t_epoch * full_epochs

            if T_min <= t_full_plan <= T_max:
                # Full data lands within the tolerance band
                epochs = full_epochs
                budget = -1
                est    = t_full_plan
            elif t_full_plan < T_min:
                # Client finishes too early → add 1 epoch and chunk to T_target.
                # full_epochs < base_epochs here, so full_epochs+1 ≤ base_epochs.
                epochs = full_epochs + 1
                budget = max(min_samples, int(T_target / epochs / t1k * 1000))
                est    = t1k * budget / 1000.0 * epochs
            else:
                # Even 1 full epoch is too slow → 1 epoch, chunk to T_max
                epochs = 1
                budget = max(min_samples, int(T_max / t1k * 1000))
                est    = t1k * budget / 1000.0

        result[cid] = {
            "local_epochs":  epochs,
            "sample_budget": budget,
            "est_time_sec":  round(est, 2),
        }

    return result


def make_adaptive_log(
    params: Dict[int, Dict],
    profiles: Dict[int, Dict[str, float]],
    base_epochs: int,
    mode: str,
    tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Build a JSON-serializable summary of the adaptive schedule for cluster_profile.json.

    Includes T_target, which client defines it, T_min/T_max, and per-client schedule.
    """
    tpe = _t_per_epoch(profiles)
    if not tpe:
        return {}

    T_target     = min(t * base_epochs for t in tpe.values())
    baseline_cid = min(tpe, key=lambda c: tpe[c])  # fastest client sets T_target

    return {
        "mode":               mode,
        "tolerance":          tolerance,
        "base_epochs":        base_epochs,
        "T_target_sec":       round(T_target, 2),
        "T_min_sec":          round(T_target * (1 - tolerance), 2),
        "T_max_sec":          round(T_target * (1 + tolerance), 2),
        "baseline_client_id": baseline_cid,
        "clients": {
            str(cid): {
                "local_epochs":  p["local_epochs"],
                "sample_budget": p["sample_budget"],
                "est_time_sec":  p["est_time_sec"],
            }
            for cid, p in sorted(params.items())
        },
    }


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
    mode: str = "maximize-epochs",
    tolerance: float = 0.10,
) -> None:
    """Print per-client adaptive training schedule to stdout."""
    tpe = _t_per_epoch(profiles)
    T_target     = min(t * base_epochs for t in tpe.values()) if tpe else 0.0
    T_min        = T_target * (1.0 - tolerance)
    T_max        = T_target * (1.0 + tolerance)
    baseline_cid = min(tpe, key=lambda c: tpe[c]) if tpe else -1

    width = 84
    print("=" * width)
    print(
        f"  Адаптивное расписание  "
        f"(mode={mode}, base_epochs={base_epochs}, "
        f"T_target={T_target:.1f}s [{T_min:.1f}s – {T_max:.1f}s], tol={int(tolerance*100)}%)"
    )
    print("=" * width)
    print(f"  {'ID':>3}  {'Скорость':>10}  {'Сэмплов':>8}  {'Эпох':>5}  {'Чанк':>8}  {'Оценка':>8}  {'':>8}")
    print("-" * width)

    for cid in sorted(params):
        p      = params[cid]
        m      = profiles.get(cid, {})
        sps    = m.get("bench_samples_per_sec", 0.0)
        n      = int(m.get("data_num_samples", 0))
        epochs = p["local_epochs"]
        budget = p["sample_budget"]
        est    = p["est_time_sec"]

        budget_str = "все" if budget == -1 else str(budget)
        est_str    = f"{est:.1f}s" if est != -1.0 else "?"

        if est == -1.0:
            note = "?"
        elif est <= T_min * 0.999:
            note = "< T_min"
        elif est <= T_target * 1.001:
            note = "✓" + (" ★" if cid == baseline_cid else "")
        elif est <= T_max * 1.001:
            note = "≈"
        else:
            note = "slow"

        print(
            f"  {cid:>3}  {sps:>8.0f}/s  {n:>8}  {epochs:>5}  {budget_str:>8}  {est_str:>8}  {note:<8}"
        )

    print("=" * width)
