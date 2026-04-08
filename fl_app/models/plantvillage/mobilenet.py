import timm
import torch.nn as nn


def build_mobilenetv3_small(num_classes: int = 38) -> nn.Module:
    """MobileNetV3-Small с предобученными весами ImageNet для PlantVillage.

    Вход: 96×96 (в 2.3× меньше нативного 224×224).
    Feature maps через blocks: 24→12→6→6→3→3→GAP — без stride-фиксов.
    2-phase fine-tuning: фаза 1 — только classifier, фаза 2 — весь backbone с BN frozen.
    """
    return timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.2,
    )
