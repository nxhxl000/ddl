from __future__ import annotations
from pathlib import Path
import csv, time

class CsvLogger:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "metrics.csv"
        self._t0 = None
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["round","val_acc","val_loss","fit_loss","fit_acc","time_sec"])

    def on_round_begin(self, rnd: int):
        if self._t0 is None:
            self._t0 = time.perf_counter()
        self._r0 = time.perf_counter()

    def on_round_end(self, rnd: int, val_acc: float | None, val_loss: float | None,
                     fit_loss: float | None, fit_acc: float | None):
        elapsed = time.perf_counter() - self._t0
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rnd, val_acc, val_loss, fit_loss, fit_acc, round(elapsed, 3)])

    def path_csv(self) -> Path:
        return self.path