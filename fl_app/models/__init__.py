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


from fl_app.models.cifar100 import WideResNet
from fl_app.models.cifar100.simple_cnn import SimpleCNN
from fl_app.models.plantvillage.efficientnet import (
    build_efficientnet_b0_scratch,
)

# ── Per-model hyperparameters ─────────────────────────────────────────────────
#
# lr снижен с 0.1 (централизованное) до 0.01 (FL): клиент делает 3 эпохи,
# высокий lr → сильный client drift.
# batch_size=64: CPU-клиенты + малые партиции (~5-8K сэмплов).
# Scheduler не используется (3 эпохи на клиенте — бесполезен).

_WRN_28_4_HPARAMS = TrainHParams(
    lr=0.01,
    batch_size=64,
    momentum=0.9,
    weight_decay=5e-4,
    num_workers=0,
)

_WRN_16_4_HPARAMS = TrainHParams(
    lr=0.01,
    batch_size=64,
    momentum=0.9,
    weight_decay=5e-4,
    num_workers=0,
)

_SIMPLE_CNN_HPARAMS = TrainHParams(
    lr=0.01,
    batch_size=64,
    momentum=0.9,
    weight_decay=1e-4,
    num_workers=0,
)

_EFFNET_B0_HPARAMS = TrainHParams(
    lr=1e-3,
    batch_size=32,
    momentum=0.0,        # AdamW, не используется
    weight_decay=1e-2,
    num_workers=0,
)

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "wrn_28_4": ModelConfig(
        cls=WideResNet,
        kwargs={"depth": 28, "widen": 4, "num_classes": 100, "drop_rate": 0.3},
        hparams=_WRN_28_4_HPARAMS,
    ),
    "wrn_16_4": ModelConfig(
        cls=WideResNet,
        kwargs={"depth": 16, "widen": 4, "num_classes": 100, "drop_rate": 0.3},
        hparams=_WRN_16_4_HPARAMS,
    ),
    "simple_cnn": ModelConfig(
        cls=SimpleCNN,
        kwargs={"num_classes": 100, "drop_rate": 0.25},
        hparams=_SIMPLE_CNN_HPARAMS,
    ),
    "efficientnet_b0": ModelConfig(
        cls=build_efficientnet_b0_scratch,
        kwargs={"num_classes": 38},
        hparams=_EFFNET_B0_HPARAMS,
    ),
}


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
