import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(channels: int, use_group_norm: bool = False) -> nn.Module:
    """BatchNorm2d или GroupNorm(min(32, channels))."""
    if use_group_norm:
        num_groups = min(32, channels)
        return nn.GroupNorm(num_groups, channels)
    return nn.BatchNorm2d(channels)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention за минимум параметров."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        w = self.fc(w).view(x.size(0), -1, 1, 1)
        return x * w


class PreActBlock(nn.Module):
    """Pre-activation BasicBlock + SE (BN→ReLU→Conv порядок)."""
    def __init__(self, in_ch, out_ch, stride=1, use_group_norm=False):
        super().__init__()
        self.bn1   = _norm_layer(in_ch, use_group_norm)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = _norm_layer(out_ch, use_group_norm)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.se    = SEBlock(out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.se(out)
        return out + self.shortcut(x)


class CifarSEResNet(nn.Module):
    """Lightweight SE-ResNet для CIFAR-100 (32×32), FL-friendly.

    Архитектура (n=2 блока на стадию):
      stem  → 64ch, 32×32  (Conv 3×3, stride 1 — без downsampling)
      stage1 → 64ch,  32×32
      stage2 → 128ch, 16×16
      stage3 → 256ch,  8×8
      GAP → Dropout → Linear(256, 100)

    Конфиги по параметрам:
      n=2  → ~2.8M  (самый лёгкий, для слабых узлов)
      n=3  → ~4.5M  (баланс)
      n=4  → ~6.2M  ≈ WRN-28-4 по размеру

    use_group_norm=True: заменяет BatchNorm2d на GroupNorm — устойчивее
    к non-IID в федеративном обучении (BN statistics расходятся между клиентами).
    """
    def __init__(self, num_classes: int = 100, n: int = 2, drop_rate: float = 0.3,
                 use_group_norm: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            _norm_layer(64, use_group_norm),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(64,  64,  n, stride=1, use_group_norm=use_group_norm)
        self.stage2 = self._make_stage(64,  128, n, stride=2, use_group_norm=use_group_norm)
        self.stage3 = self._make_stage(128, 256, n, stride=2, use_group_norm=use_group_norm)
        self.bn_out = _norm_layer(256, use_group_norm)
        self.drop   = nn.Dropout(drop_rate)
        self.fc     = nn.Linear(256, num_classes)
        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n, stride, use_group_norm=False):
        layers = [PreActBlock(in_ch, out_ch, stride, use_group_norm=use_group_norm)]
        for _ in range(1, n):
            layers.append(PreActBlock(out_ch, out_ch, 1, use_group_norm=use_group_norm))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn_out(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.drop(out)
        return self.fc(out)
