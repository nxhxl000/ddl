"""Модели + их гиперпараметры обучения.

Архитектура + HParams (optimizer, lr, momentum, wd, bs, local_epochs)
живут здесь, рядом с моделью. Per-strategy оверрайды в `per_strategy`.

pyproject.toml отвечает только за "что запускать" (partition, model, aggregation).
Любой HParams можно переопределить в run_config: `client-lr`, `{agg}-client-lr`, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from .cifar100.se_resnet import CifarSEResNet
from .cifar100.wrn import WideResNet
from .plantvillage.efficientnet import build_efficientnet_b0_scratch


@dataclass(frozen=True)
class HParams:
    """Гиперпараметры обучения для одной модели.

    per_strategy — частичные оверрайды по имени агрегации, например:
      {"scaffold": {"client-lr": 0.3, "local-epochs": 5}}
    Ключи используют тот же kebab-case, что `_hp` в client_app.py.
    """
    optimizer: str = "sgd"
    client_lr: float = 0.03
    client_momentum: float = 0.9
    client_weight_decay: float = 5e-4
    batch_size: int = 64
    local_epochs: int = 2
    per_strategy: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[], nn.Module]
    hparams: HParams


# CIFAR-100 общие per-strategy оверрайды (тюнились в предыдущих экспериментах)
_CIFAR_PER_STRATEGY: dict[str, dict] = {
    "fedavg":   {"client-lr": 0.1,  "local-epochs": 3},
    "fedavgm":  {"client-lr": 0.05, "local-epochs": 3},
    "fedprox":  {"client-lr": 0.1,  "local-epochs": 3},
    "scaffold": {"client-lr": 0.3,  "local-epochs": 5},
    "fednova":  {"client-lr": 0.1,  "local-epochs": 3},
    "fednovam": {"client-lr": 0.05, "local-epochs": 3},
}


_MODELS: dict[str, ModelSpec] = {
    "wrn_16_4": ModelSpec(
        build=lambda: WideResNet(depth=16, widen=4, num_classes=100, drop_rate=0.3),
        hparams=HParams(
            optimizer="sgd",
            client_lr=0.03, client_momentum=0.9, client_weight_decay=5e-4,
            batch_size=64, local_epochs=2,
            per_strategy=_CIFAR_PER_STRATEGY,
        ),
    ),
    "se_resnet": ModelSpec(
        build=lambda: CifarSEResNet(num_classes=100, n=2, drop_rate=0.3),
        hparams=HParams(
            optimizer="sgd",
            client_lr=0.03, client_momentum=0.9, client_weight_decay=5e-4,
            batch_size=64, local_epochs=2,
            per_strategy=_CIFAR_PER_STRATEGY,
        ),
    ),
    "effnet_b0": ModelSpec(
        build=lambda: build_efficientnet_b0_scratch(num_classes=38),
        hparams=HParams(
            optimizer="adamw",
            client_lr=1e-3, client_weight_decay=1e-2,
            batch_size=32, local_epochs=2,
            per_strategy={},  # пока не тюнили
        ),
    ),
}


def build_model(name: str) -> nn.Module:
    key = name.strip().lower()
    if key not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_MODELS)}")
    return _MODELS[key].build()


def get_hparams(model_name: str, aggregation: str) -> dict:
    """Resolve HParams → dict (kebab-case keys) с учётом per-strategy оверрайдов.

    Возвращает:
      optimizer, client-lr, client-momentum, client-weight-decay,
      batch-size, local-epochs
    """
    key = model_name.strip().lower()
    if key not in _MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Available: {sorted(_MODELS)}")
    hp = _MODELS[key].hparams
    base = {
        "optimizer":           hp.optimizer,
        "client-lr":           hp.client_lr,
        "client-momentum":     hp.client_momentum,
        "client-weight-decay": hp.client_weight_decay,
        "batch-size":          hp.batch_size,
        "local-epochs":        hp.local_epochs,
    }
    base.update(hp.per_strategy.get(aggregation.strip().lower(), {}))
    return base
