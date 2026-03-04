from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Training hyperparameters ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainHParams:
    lr:           float          # шаг градиентного спуска (SGD)
    batch_size:   int            # образцов в одном батче
    momentum:     float = 0.9   # момент SGD (сглаживает осцилляции)
    weight_decay: float = 0.0   # L2-регуляризация (0 = выключена)
    num_workers:  int   = 0     # потоки DataLoader (0 = основной поток)


# ── Per-model hyperparameters ─────────────────────────────────────────────────
#
# local_epochs задаётся в pyproject.toml (ключ "local-epochs")
# и применяется одинаково для всех клиентов в раунде.

# CifarCNN — double-conv CNN с BatchNorm и Dropout для CIFAR-10 (RGB, 32×32)
# lr           = 0.01   — подобрано локально; BatchNorm позволяет тот же lr
# batch_size   = 64     — хорошо вмещается в RAM клиентов
# momentum     = 0.9    — стандарт для SGD
# weight_decay = 1e-4   — лёгкая L2-регуляризация (снижает overfit)
# num_workers  = 2      — параллельная загрузка данных
_CIFARCNN_HPARAMS = TrainHParams(
    lr=0.01,
    batch_size=64,
    momentum=0.9,
    weight_decay=1e-4,
    num_workers=0,
)

# LeNet-5 — классическая CNN для MNIST (grayscale, 28×28)
# lr           = 0.01   — более высокий lr: MNIST проще, быстрее сходится
# batch_size   = 64     — то же
# momentum     = 0.9    — то же
# weight_decay = 0.0    — MNIST не требует регуляризации
# num_workers  = 2      — то же
_LENET5_HPARAMS = TrainHParams(
    lr=0.01,
    batch_size=64,
    momentum=0.9,
    weight_decay=0.0,
    num_workers=0,
)


# ── Model architectures ───────────────────────────────────────────────────────

class CifarCNN(nn.Module):
    """Double-conv CNN с BatchNorm и Dropout для CIFAR-10 (in_channels=3). ~141K параметров.

    Block 1: Conv(3→32)+BN+ReLU + Conv(32→32)+BN+ReLU + MaxPool(2)  → 32×16×16
    Block 2: Conv(32→64)+BN+ReLU + Conv(64→64)+BN+ReLU + MaxPool(2)  → 64×8×8
    Block 3: Conv(64→128)+BN+ReLU + GAP                               → 128×1×1
    Classifier: Dropout(0.5) + Linear(128→num_classes)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x).flatten(1)
        return self.classifier(x)


class LeNet5(nn.Module):
    """Классический LeNet-5 для MNIST (in_channels=1, 28×28). ~60K параметров."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool  = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    cls:     type
    kwargs:  Dict[str, Any]
    hparams: TrainHParams


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "cifarcnn": ModelConfig(
        cls=CifarCNN,
        kwargs={"in_channels": 3, "num_classes": 10},
        hparams=_CIFARCNN_HPARAMS,
    ),
    "lenet5": ModelConfig(
        cls=LeNet5,
        kwargs={"num_classes": 10},
        hparams=_LENET5_HPARAMS,
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
