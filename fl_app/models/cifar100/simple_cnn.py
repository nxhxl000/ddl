import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Лёгкий CNN для CIFAR-100 (32x32). ~210K параметров.

    Для быстрого тестирования пайплайна и экспериментов на CPU.
    Потолок accuracy: ~45-55% (CIFAR-100 сложный датасет).
    """

    def __init__(self, num_classes=100, drop_rate=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 64, 32x32 → 16x16
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop_rate),

            # Block 2: 64 → 128, 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop_rate),

            # Block 3: 128 → 256, 8x8 → 4x4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
