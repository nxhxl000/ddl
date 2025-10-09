from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np

def split_iid(num_samples: int, num_clients: int, seed: int = 42) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [s.astype(int).tolist() for s in splits]

def split_dirichlet(labels: np.ndarray, num_clients: int, alpha: float,
                    num_classes: int, seed: int = 42) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    clients: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = class_indices[c]
        n_c = len(idx_c)
        props = rng.dirichlet([alpha] * num_clients)
        counts = (np.round(props * n_c)).astype(int)

        # корректировка суммы на случай округления
        diff = n_c - counts.sum()
        if diff != 0:
            order = np.argsort(props)[::-1] if diff > 0 else np.argsort(props)
            for k in order[:abs(diff)]:
                counts[k] += 1 if diff > 0 else -1

        start = 0
        for k, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            part = idx_c[start:start+cnt]
            clients[k].extend(part.tolist())
            start += cnt

    for k in range(num_clients):
        rng.shuffle(clients[k])

    return clients

def save_split(path: Path, split: List[List[int]], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"meta": meta, "clients": split}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_split(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)