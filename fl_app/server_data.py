"""fl_app/server_data.py — Server dataset scheduling for heterogeneity reduction.

Distributes the balanced server dataset to clients to reduce mean_pairwise_JS.

Algorithm — Proportional-Deficit:
  For each class k, distribute the available budget (per_class samples)
  proportionally to each client's deficit from uniform:

      deficit_i[k] = max(0, 1/K - P_i[k])
      s_i[k] = per_class * deficit_i[k] / sum_j(deficit_j[k])   [exclusive]
      s_i[k] = per_class * deficit_i[k] / sum_k(deficit_i[k])   [shared, per-client budget]

  where per_class = server_size // num_classes.

  Properties:
    - exclusive: sum_i(s_i[k]) = per_class by construction (no extra scaling needed)
    - both modes: JS always decreases (monotone improvement regardless of budget size)
    - small budgets: 3x more JS reduction vs exact-uniform at same server_size

Two distribution modes:
  shared:    Each client independently gets per_class samples total, distributed
             across their deficit classes. Same sample may go to multiple clients.
  exclusive: Budget per_class is split across clients for each class.
             No client can take more than their proportional share.

Disabled when server/ partition does not exist or manifest.server_size is None.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core allocation
# ─────────────────────────────────────────────────────────────────────────────

def compute_server_schedule(
    profiles: Dict[int, Dict],
    num_classes: int,
    server_size: int,
    mode: str = "shared",
) -> Dict[int, List[int]]:
    """Compute per-client per-class server sample counts (Proportional-Deficit algorithm).

    Args:
        profiles:    {partition_id: metrics_dict} from run_profiling_round().
        num_classes: Total number of classes (from manifest).
        server_size: Total samples in the server dataset (= per_class * K).
        mode:        "exclusive" — per_class budget split across clients per class
                                   (sum_i s_i[k] = per_class by construction);
                     "shared"    — each client gets per_class total, distributed
                                   across their own deficit classes (samples reused).

    Returns:
        {partition_id: [count_class_0, ..., count_class_{K-1}]}
        Zero counts mean no server samples for that class.
    """
    if mode not in ("shared", "exclusive"):
        raise ValueError(f"Unknown server mode: {mode!r}. Use 'shared' or 'exclusive'.")

    per_class = server_size // num_classes
    uniform   = 1.0 / num_classes
    K         = num_classes

    # Build per-client deficit vectors
    cids = sorted(profiles.keys())
    P: Dict[int, List[float]] = {}
    for cid in cids:
        m   = profiles[cid]
        n_i = int(m.get("data_num_samples", 0))
        if n_i > 0:
            P[cid] = [int(m.get(f"data_cls_{k}", 0)) / n_i for k in range(K)]
        else:
            P[cid] = [uniform] * K

    # deficit_i[k] = max(0, 1/K - P_i[k])
    deficit: Dict[int, List[float]] = {
        cid: [max(0.0, uniform - P[cid][k]) for k in range(K)]
        for cid in cids
    }

    schedule: Dict[int, List[int]] = {cid: [0] * K for cid in cids}

    if mode == "exclusive":
        # For each class k: distribute per_class proportionally to deficit_i[k]
        # sum_i(s_i[k]) = per_class by construction → no extra enforcement needed
        for k in range(K):
            total_deficit = sum(deficit[cid][k] for cid in cids)
            if total_deficit <= 0:
                continue
            for cid in cids:
                schedule[cid][k] = int(per_class * deficit[cid][k] / total_deficit)

    else:  # shared
        # Each client independently gets per_class samples total,
        # distributed across their deficit classes proportionally.
        for cid in cids:
            total_deficit = sum(deficit[cid])
            if total_deficit <= 0:
                continue
            for k in range(K):
                if deficit[cid][k] > 0:
                    schedule[cid][k] = int(per_class * deficit[cid][k] / total_deficit)

    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneity metric after server data redistribution
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_js(
    schedule: Dict[int, List[int]],
    profiles: Dict[int, Dict],
    num_classes: int,
) -> float:
    """Compute mean_pairwise_js of effective distributions Q_i after adding server samples.

    Q_i[k] = (n_i * P_i[k] + s_i[k]) / (n_i + sum_k(s_i[k]))

    Returns mean JS divergence across all client pairs (same metric as profiling.py).
    """
    import math

    uniform = 1.0 / num_classes

    def _js(p: List[float], q: List[float]) -> float:
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
        kl = lambda a, b: sum(
            ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 1e-12 and bi > 1e-12
        )
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)  # JS in nats, matches profiling.py

    # Build effective distributions Q_i
    Q: Dict[int, List[float]] = {}
    for cid, counts in schedule.items():
        m   = profiles.get(cid, {})
        n_i = int(m.get("data_num_samples", 0))
        if n_i == 0:
            Q[cid] = [uniform] * num_classes
            continue
        P_i   = [int(m.get(f"data_cls_{k}", 0)) / n_i for k in range(num_classes)]
        total = n_i + sum(counts)
        Q[cid] = [
            (n_i * P_i[k] + counts[k]) / total if total > 0 else uniform
            for k in range(num_classes)
        ]

    cids = sorted(Q.keys())
    if len(cids) < 2:
        return 0.0

    pairs = [(cids[i], cids[j]) for i in range(len(cids)) for j in range(i + 1, len(cids))]
    return sum(_js(Q[a], Q[b]) for a, b in pairs) / len(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# ConfigRecord encoding
# ─────────────────────────────────────────────────────────────────────────────

def to_server_config_dict(schedule: Dict[int, List[int]]) -> Dict[str, float]:
    """Flatten server schedule into ConfigRecord-compatible flat dict.

    Keys: c{pid}_srv_{k} — count of class-k samples for client pid.
    The round seed is NOT included; clients derive it from server-round * 997.
    """
    flat: Dict[str, float] = {}
    for pid, counts in schedule.items():
        for k, cnt in enumerate(counts):
            flat[f"c{pid}_srv_{k}"] = float(cnt)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Client-side: sample from local server/ partition
# ─────────────────────────────────────────────────────────────────────────────

def sample_server_dataset(
    server_path: Path | str,
    class_counts: List[int],
    round_seed: int,
    *,
    label_col: str = "label",
):
    """Sample from the local server/ partition according to per-class counts.

    Args:
        server_path:  Path to the local server/ HuggingFace dataset directory.
        class_counts: [count_class_0, ..., count_class_{K-1}].
        round_seed:   Seed for deterministic per-round shuffling within each class.
        label_col:    Label column name.

    Returns:
        HuggingFace Dataset with sum(class_counts) samples.
    """
    from datasets import load_from_disk

    server_ds = load_from_disk(str(server_path))
    labels    = server_ds[label_col]

    class_indices: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(i)

    selected: List[int] = []
    for k, count in enumerate(class_counts):
        if count <= 0 or k not in class_indices:
            continue
        idxs = list(class_indices[k])
        rng  = random.Random(round_seed + k * 100)
        rng.shuffle(idxs)
        selected.extend(idxs[:count])

    if not selected:
        return server_ds.select([])

    rng = random.Random(round_seed)
    rng.shuffle(selected)
    return server_ds.select(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_server_log(
    schedule: Dict[int, List[int]],
    profiles: Dict[int, Dict],
    num_classes: int,
    server_size: int,
    mode: str,
    mean_pairwise_js_before: float = 0.0,
) -> Dict:
    """Build JSON-serialisable summary for cluster_profile.json."""
    per_class = server_size // num_classes
    uniform   = 1.0 / num_classes

    clients_log = {}
    for cid in sorted(schedule):
        m      = profiles.get(cid, {})
        n_i    = int(m.get("data_num_samples", 0))
        P_i    = (
            [int(m.get(f"data_cls_{k}", 0)) / n_i for k in range(num_classes)]
            if n_i > 0 else [uniform] * num_classes
        )
        counts = schedule[cid]
        total  = sum(counts)
        Q_i    = [
            (n_i * P_i[k] + counts[k]) / (n_i + total) if (n_i + total) > 0 else 0.0
            for k in range(num_classes)
        ]
        clients_log[str(cid)] = {
            "class_counts":         counts,
            "total_server_samples": total,
            "effective_dist":       [round(q, 4) for q in Q_i],
        }

    js_after = compute_effective_js(schedule, profiles, num_classes)
    reduction = (
        (mean_pairwise_js_before - js_after) / mean_pairwise_js_before * 100
        if mean_pairwise_js_before > 1e-9 else 0.0
    )

    return {
        "mode":                    mode,
        "server_size":             server_size,
        "per_class":               per_class,
        "mean_pairwise_js_before": round(mean_pairwise_js_before, 6),
        "mean_pairwise_js_after":  round(js_after, 6),
        "js_reduction_pct":        round(reduction, 2),
        "clients":                 clients_log,
    }


def print_server_schedule_summary(
    schedule: Dict[int, List[int]],
    profiles: Dict[int, Dict],
    num_classes: int,
    server_size: int,
    mode: str,
    class_names: Optional[List[str]] = None,
    mean_pairwise_js_before: float = 0.0,
) -> None:
    """Print per-client server dataset allocation and JS heterogeneity before/after."""
    per_class = server_size // num_classes

    width = 84
    print("=" * width)
    print(
        f"  Серверный датасет  "
        f"(mode={mode}, server_size={server_size}, {per_class}/class)"
    )
    print("=" * width)
    print(f"  {'ID':>3}  {'Сэмплов':>8}  {'Топ-3 классов (дефицит → сервер)':}")
    print("-" * width)

    for cid in sorted(schedule):
        counts   = schedule[cid]
        total    = sum(counts)
        top3     = sorted(range(num_classes), key=lambda k: -counts[k])[:3]
        top3_str = "  ".join(
            f"{class_names[k] if class_names else k}:{counts[k]}"
            for k in top3
            if counts[k] > 0
        )
        print(f"  {cid:>3}  {total:>8}  {top3_str}")

    if mode == "exclusive":
        print("-" * width)
        total_per_class = [
            sum(schedule[cid][k] for cid in schedule)
            for k in range(num_classes)
        ]
        max_util = max(total_per_class) if total_per_class else 0
        print(
            f"  Макс. нагрузка на класс: {max_util}/{per_class} "
            f"({max_util / per_class * 100:.0f}%)"
        )

    # ── Heterogeneity before / after ─────────────────────────────────────────
    js_after = compute_effective_js(schedule, profiles, num_classes)
    reduction = (
        (mean_pairwise_js_before - js_after) / mean_pairwise_js_before * 100
        if mean_pairwise_js_before > 1e-9 else 0.0
    )
    print("-" * width)
    if mean_pairwise_js_before > 1e-9:
        print(
            f"  Гетерогенность (mean pairwise JS):  "
            f"до = {mean_pairwise_js_before:.4f}  →  "
            f"после = {js_after:.4f}  "
            f"({reduction:+.1f}%)"
        )
    else:
        print(f"  Гетерогенность после сервер-данных: JS = {js_after:.4f}")
    print("=" * width)
