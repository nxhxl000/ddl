from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn


# ── Training hyperparameters ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainHParams:
    lr:           float          # шаг градиентного спуска (SGD)
    batch_size:   int            # образцов в одном батче
    momentum:     float = 0.9   # момент SGD (сглаживает осцилляции)
    weight_decay: float = 0.0   # L2-регуляризация (0 = выключена)
    num_workers:  int   = 0     # потоки DataLoader (0 = основной поток)


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    cls:     type
    kwargs:  Dict[str, Any]
    hparams: TrainHParams


MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def build_model(name: str) -> nn.Module:
    """Создать модель по имени из реестра."""
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[key]
    return cfg.cls(**cfg.kwargs)


def get_hparams(name: str) -> TrainHParams:
    """Получить гиперпараметры обучения для модели по имени."""
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key].hparams
