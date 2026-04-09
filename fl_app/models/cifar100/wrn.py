import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(num_channels: int, use_group_norm: bool = False) -> nn.Module:
    """BatchNorm2d или GroupNorm в зависимости от режима."""
    if use_group_norm:
        num_groups = min(32, num_channels // 4) if num_channels >= 4 else 1
        return nn.GroupNorm(num_groups, num_channels)
    return nn.BatchNorm2d(num_channels)


class WideBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop_rate, use_group_norm=False):
        super().__init__()
        self.bn1   = _norm_layer(in_ch, use_group_norm)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = _norm_layer(out_ch, use_group_norm)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut  = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """WideResNet для CIFAR (32×32).

    Рекомендуемые конфиги:
      depth=28, widen=4  → ~5.9M params  (FL-пригодный)
      depth=28, widen=10 → ~36M params   (reference ceiling)

    Обучение: SGD + Nesterov, lr=0.1, MultiStepLR([30,60,80], gamma=0.2), 100 эпох.
    """
    def __init__(self, depth=28, widen=4, num_classes=100, drop_rate=0.3,
                 use_group_norm=False):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n  = (depth - 4) // 6
        ch = [16, 16 * widen, 32 * widen, 64 * widen]
        self._use_gn = use_group_norm

        self.conv0  = nn.Conv2d(3, ch[0], 3, padding=1, bias=False)
        self.group1 = self._group(ch[0], ch[1], n, stride=1, drop_rate=drop_rate)
        self.group2 = self._group(ch[1], ch[2], n, stride=2, drop_rate=drop_rate)
        self.group3 = self._group(ch[2], ch[3], n, stride=2, drop_rate=drop_rate)
        self.bn     = _norm_layer(ch[3], use_group_norm)
        self.fc     = nn.Linear(ch[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _group(self, in_ch, out_ch, n, stride, drop_rate):
        layers = [WideBlock(in_ch, out_ch, stride, drop_rate, self._use_gn)]
        for _ in range(1, n):
            layers.append(WideBlock(out_ch, out_ch, 1, drop_rate, self._use_gn))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
