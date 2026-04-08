import timm
import torch.nn as nn


def build_efficientnet_b0(
    num_classes: int = 100,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """EfficientNet-B0 адаптированный для CIFAR-100 (32×32), обучение с нуля.

    Изменения vs оригинала:
      - conv_stem stride 2→1: не режем разрешение на первом слое (32→16 → 32→32)
      - drop_rate: dropout перед классификатором (default 0.2)
      - drop_path_rate: stochastic depth по всем блокам (default 0.2) — ключевой
        регуляризатор для EfficientNet; без него модель переобучается на CIFAR-100

    Итоговое разрешение через сеть (32→32→16→8→4→4→2→2→GAP).
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    # Stem: stride 2 → 1 (32×32 → 32×32)
    model.conv_stem.stride = (1, 1)

    # blocks[3][0]: stride 2 → 1 (8×8 → 8×8 вместо 8→4)
    # Без этого GAP работает по сетке 2×2 — слишком мало для 100 классов.
    # С этим фиксом: GAP по 4×4, как в ResNet/WRN для CIFAR.
    # Безопасно: stride-2 блоки в EfficientNet не имеют identity shortcut (has_shortcut=False).
    model.blocks[3][0].conv_dw.stride = (1, 1)

    return model
