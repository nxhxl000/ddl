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
    mode: str                       # "none" | "chunk" | "epochs"
    target: str                     # "min" | "median"
    T_target: float                 # секунды
    T_upper: float                  # T_target * (1 + tolerance)
    tolerance: float
    base_epochs: int
    chunks: dict[int, float] = field(default_factory=dict)
    epochs: dict[int, int]   = field(default_factory=dict)

    def _num_pids(self) -> int:
        return (max(self.chunks.keys()) + 1) if self.chunks else 0

    def chunks_str(self) -> str:
        """'c0,c1,c2,...' для per-client-chunks (positional, по pid)."""
        n = self._num_pids()
        return ",".join(f"{self.chunks.get(p, 1.0):.4f}" for p in range(n)) if n else ""

    def epochs_str(self) -> str:
        n = self._num_pids()
        return ",".join(f"{self.epochs.get(p, self.base_epochs)}" for p in range(n)) if n else ""

    def to_dict(self) -> dict:
        return {
            "mode":        self.mode,
            "target":      self.target,
            "T_target":    self.T_target,
            "T_upper":     self.T_upper,
            "tolerance":   self.tolerance,
            "base_epochs": self.base_epochs,
            "chunks":      {str(k): v for k, v in sorted(self.chunks.items())},
            "epochs":      {str(k): v for k, v in sorted(self.epochs.items())},
        }


def compute_schedule(
    t_compute_by_pid: dict[int, float],
    *,
    mode: str,
    base_epochs: int,
    target: str = "min",
    tolerance: float = 0.05,
    min_chunk: float = 0.1,
    min_epochs: int = 1,
) -> Schedule:
    """Из round-1 таймингов → per-client chunks/epochs.

    target=min: T_target = min(times); target=median: median.
    Клиенты с t_compute ≤ T_target*(1+tolerance) не throttle'ятся.
    """
    if mode not in ("none", "chunk", "epochs"):
        raise ValueError(f"Unknown mode: {mode!r}")

    times = list(t_compute_by_pid.values())
    if not times:
        return Schedule(
            mode=mode, target=target, T_target=0.0, T_upper=0.0,
            tolerance=tolerance, base_epochs=base_epochs,
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
    chunks: dict[int, float] = {}
    epochs: dict[int, int]   = {}

    for pid, t in t_compute_by_pid.items():
        # В полосе, mode=none или невалидное t — оставляем full work
        if mode == "none" or t <= 0 or t <= T_upper:
            chunks[pid] = 1.0
            epochs[pid] = base_epochs
            continue

        ratio = T_upper / t  # < 1 для медленных
        if mode == "chunk":
            chunks[pid] = max(min_chunk, ratio)
            epochs[pid] = base_epochs
        else:  # mode == "epochs"
            chunks[pid] = 1.0
            epochs[pid] = max(min_epochs, round(base_epochs * ratio))

    return Schedule(
        mode=mode, target=target, T_target=T_target, T_upper=T_upper,
        tolerance=tolerance, base_epochs=base_epochs,
        chunks=chunks, epochs=epochs,
    )
