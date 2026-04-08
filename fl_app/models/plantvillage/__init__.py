"""Модели для PlantVillage (224×224, 38 классов).

Использование:
    from fl_app.models.plantvillage import build_model, MODEL_CONFIGS
    model  = build_model("efficientnet_b0")
    config = MODEL_CONFIGS["efficientnet_b0"]
"""
import torch.nn as nn
from .efficientnet import build_efficientnet_b0_pretrained, build_efficientnet_b0_scratch
from .mobilenet    import build_mobilenetv3_small

MODEL_CONFIGS: dict = {
    "mobilenetv3_small": dict(
        phase1_epochs  = 10,
        lr_head_p1     = 1e-3,
        phase2_epochs  = 30,
        lr_backbone    = 1e-5,
        lr_head_p2     = 1e-4,
        weight_decay   = 1e-4,
        batch_size     = 128,      # мелкие изображения → больший батч
        mixup_alpha    = 0.0,
        label_smooth   = 0.1,
        image_size     = 96,
        description    = "MobileNetV3-Small (pretrained ImageNet), 96×96, ~1.6M params",
    ),
    "efficientnet_b0_scratch": dict(
        pretrained     = False,
        optimizer      = "adamw",
        lr             = 1e-3,
        weight_decay   = 1e-2,
        scheduler      = "cosine",
        epochs         = 100,
        batch_size     = 64,
        mixup_alpha    = 0.4,
        label_smooth   = 0.1,
        image_size     = 224,
        description    = "EfficientNet-B0 (from scratch), ~4.0M params",
    ),
    "efficientnet_b0": dict(
        # Фаза 1: замораживаем backbone, обучаем только голову
        phase1_epochs  = 10,
        lr_head_p1     = 1e-3,     # высокий LR для новой головы
        # Фаза 2: размораживаем, BN остаётся замороженным
        phase2_epochs  = 30,
        lr_backbone    = 1e-5,     # очень низкий — не перезаписываем ImageNet-признаки
        lr_head_p2     = 1e-4,     # голова тоже снижается
        weight_decay   = 1e-4,
        batch_size     = 64,
        mixup_alpha    = 0.0,      # Mixup отключён: смесь текстур болезней нереалистична
        label_smooth   = 0.1,
        image_size     = 224,
        description    = "EfficientNet-B0 (pretrained ImageNet), 2-phase fine-tune, ~4.0M params",
    ),
}


def build_model(name: str, num_classes: int = 38, **kwargs) -> nn.Module:
    if name == "efficientnet_b0":
        return build_efficientnet_b0_pretrained(num_classes=num_classes)
    elif name == "mobilenetv3_small":
        return build_mobilenetv3_small(num_classes=num_classes)
    elif name == "efficientnet_b0_scratch":
        return build_efficientnet_b0_scratch(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
