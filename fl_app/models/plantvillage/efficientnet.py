import timm
import torch.nn as nn


def build_efficientnet_b0_pretrained(num_classes: int = 38) -> nn.Module:
    """EfficientNet-B0 с предобученными весами ImageNet для PlantVillage.

    Вход: 224×224. 2-phase fine-tuning: сначала голова, затем весь backbone с замороженным BN.
    """
    return timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.3,
        drop_path_rate=0.2,
    )


def build_efficientnet_b0_scratch(num_classes: int = 38) -> nn.Module:
    """EfficientNet-B0 обучение с нуля для PlantVillage (224×224).

    Без предобученных весов — stride-фиксы не нужны (вход 224×224 нативный).
    Регуляризация: drop_rate=0.2, drop_path_rate=0.2.
    Обучение: AdamW lr=1e-3, wd=1e-2, CosineAnnealingLR, 100 эпох, Mixup=0.4.
    """
    return timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.2,
        drop_path_rate=0.2,
    )
