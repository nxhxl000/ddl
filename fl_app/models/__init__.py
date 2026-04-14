"""Модели — только архитектуры. Гиперпараметры обучения → pyproject.toml."""

from __future__ import annotations

import torch.nn as nn

from .cifar100.wrn import WideResNet
from .cifar100.se_resnet import CifarSEResNet


_MODELS: dict[str, callable] = {
    "wrn_16_4": lambda: WideResNet(depth=16, widen=4, num_classes=100, drop_rate=0.3),
    "se_resnet": lambda: CifarSEResNet(num_classes=100, n=2, drop_rate=0.3),
}


def build_model(name: str) -> nn.Module:
    key = name.strip().lower()
    if key not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_MODELS)}")
    return _MODELS[key]()
