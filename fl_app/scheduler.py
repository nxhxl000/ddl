"""Round-1 measurement → per-client schedule (chunks или epochs).

Логика: после round 1 у сервера есть t_compute_by_pid. Считаем T_target
(min или median по этим временам) + tolerance band [T_target, T_target*(1+tol)].
Клиенты в полосе не trottle'ятся; кто выше — получают пропорциональный chunk
или epochs так, чтобы попасть в T_upper.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Schedule:
    mode: str                       # "none" | "chunk" | "epochs" | "drop"
    target: str                     # "min" | "median"
    T_target: float                 # секунды
    T_upper: float                  # T_target * (1 + tolerance) — для chunk/epochs
    T_drop: float                   # T_target * (1 + drop_tolerance) — для drop
    tolerance: float
    drop_tolerance: float
    base_epochs: int
    chunks: dict[int, float] = field(default_factory=dict)
    epochs: dict[int, int]   = field(default_factory=dict)
    excluded: list[int]      = field(default_factory=list)   # pids для drop-mode

    def _num_pids(self) -> int:
        return (max(self.chunks.keys()) + 1) if self.chunks else 0

    def chunks_str(self) -> str:
        """'c0,c1,c2,...' для per-client-chunks (positional, по pid)."""
        n = self._num_pids()
        return ",".join(f"{self.chunks.get(p, 1.0):.4f}" for p in range(n)) if n else ""

    def epochs_str(self) -> str:
        n = self._num_pids()
        return ",".join(f"{self.epochs.get(p, self.base_epochs)}" for p in range(n)) if n else ""

    def excluded_str(self) -> str:
        return ",".join(str(p) for p in sorted(self.excluded))

    def to_dict(self) -> dict:
        return {
            "mode":           self.mode,
            "target":         self.target,
            "T_target":       self.T_target,
            "T_upper":        self.T_upper,
            "T_drop":         self.T_drop,
            "tolerance":      self.tolerance,
            "drop_tolerance": self.drop_tolerance,
            "base_epochs":    self.base_epochs,
            "chunks":         {str(k): v for k, v in sorted(self.chunks.items())},
            "epochs":         {str(k): v for k, v in sorted(self.epochs.items())},
            "excluded":       sorted(self.excluded),
        }


def compute_schedule(
    t_compute_by_pid: dict[int, float],
    *,
    mode: str,
    base_epochs: int,
    target: str = "min",
    tolerance: float = 0.05,
    drop_tolerance: float = 0.5,
    max_dropped: int = 3,
    min_chunk: float = 0.1,
    min_epochs: int = 1,
) -> Schedule:
    """Из round-1 таймингов → per-client schedule.

    target=min: T_target = min(times); target=median: median.

    chunk/epochs mode: клиенты с t ≤ T_target*(1+tolerance) не throttle'ятся,
        выше — chunk_fraction или local_epochs урезаются пропорционально.
    drop mode: до max_dropped самых медленных клиентов с t > T_target*(1+drop_tolerance)
        исключаются (zero reply, weight=0 в FedAvg). Остальные — full work.
    """
    if mode not in ("none", "chunk", "epochs", "drop"):
        raise ValueError(f"Unknown mode: {mode!r}")

    times = list(t_compute_by_pid.values())
    if not times:
        return Schedule(
            mode=mode, target=target, T_target=0.0, T_upper=0.0, T_drop=0.0,
            tolerance=tolerance, drop_tolerance=drop_tolerance, base_epochs=base_epochs,
        )

    if target == "min":
        T_target = min(times)
    elif target == "median":
        st = sorted(times)
        T_target = st[len(st) // 2]
    else:
        try:
            T_target = float(target)
        except (TypeError, ValueError):
            raise ValueError(f"Unknown target: {target!r}") from None

    T_upper = T_target * (1.0 + tolerance)
    T_drop  = T_target * (1.0 + drop_tolerance)
    chunks: dict[int, float] = {}
    epochs: dict[int, int]   = {}
    excluded: list[int]      = []

    # Дефолтные значения для всех клиентов
    for pid, t in t_compute_by_pid.items():
        chunks[pid] = 1.0
        epochs[pid] = base_epochs

    if mode == "none" or all(t <= 0 for t in times):
        return Schedule(
            mode=mode, target=target, T_target=T_target, T_upper=T_upper, T_drop=T_drop,
            tolerance=tolerance, drop_tolerance=drop_tolerance, base_epochs=base_epochs,
            chunks=chunks, epochs=epochs, excluded=excluded,
        )

    if mode == "drop":
        # До max_dropped самых медленных клиентов выше T_drop отбрасываем
        candidates = sorted(t_compute_by_pid.items(), key=lambda kv: kv[1], reverse=True)
        for pid, t in candidates:
            if len(excluded) >= max_dropped:
                break
            if t > T_drop:
                excluded.append(pid)
        return Schedule(
            mode=mode, target=target, T_target=T_target, T_upper=T_upper, T_drop=T_drop,
            tolerance=tolerance, drop_tolerance=drop_tolerance, base_epochs=base_epochs,
            chunks=chunks, epochs=epochs, excluded=excluded,
        )

    # chunk / epochs mode
    for pid, t in t_compute_by_pid.items():
        if t <= 0 or t <= T_upper:
            continue  # full work уже выставлен
        ratio = T_upper / t  # < 1 для медленных
        if mode == "chunk":
            chunks[pid] = max(min_chunk, ratio)
        else:  # mode == "epochs"
            epochs[pid] = max(min_epochs, round(base_epochs * ratio))

    return Schedule(
        mode=mode, target=target, T_target=T_target, T_upper=T_upper, T_drop=T_drop,
        tolerance=tolerance, drop_tolerance=drop_tolerance, base_epochs=base_epochs,
        chunks=chunks, epochs=epochs, excluded=excluded,
    )
