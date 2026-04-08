"""Модели для CIFAR-100 (32×32, scratch).

Использование:
    from fl_app.models.cifar100 import build_model, MODEL_CONFIGS
    model  = build_model("wrn_28_4")
    config = MODEL_CONFIGS["wrn_28_4"]   # дефолтные гиперпараметры
"""
from .wrn          import WideResNet
from .efficientnet import build_efficientnet_b0
from .cct          import CCT
from .se_resnet    import CifarSEResNet
from .simple_cnn   import SimpleCNN

# Дефолтные гиперпараметры обучения для каждой модели.
# Взяты из оригинальных статей / best practices для CIFAR-100 scratch.
MODEL_CONFIGS: dict = {
    "wrn_28_4": dict(
        optimizer    = "sgd",
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 5e-4,
        scheduler    = "multistep",
        milestones   = [30, 60, 80],
        gamma        = 0.2,
        #scheduler    = "cosine",
        epochs       = 100,
        batch_size   = 256,
        mixup_alpha  = 0.4,
        label_smooth = 0.1,
        description  = "WideResNet-28-4, ~5.9M params, FL-пригодный",
    ),
    "efficientnet_b0": dict(
        optimizer    = "adamw",
        lr           = 1e-3,
        weight_decay = 1e-2,   # AdamW требует сильный wd (1e-4 — почти ничего)
        scheduler    = "cosine",
        epochs       = 100,
        batch_size   = 128,
        mixup_alpha  = 0.4,
        label_smooth = 0.1,
        description  = "EfficientNet-B0 (stride fix + stochastic depth), ~5.3M params",
    ),
    "se_resnet": dict(
        optimizer    = "sgd",
        lr           = 0.1,
        momentum     = 0.9,
        weight_decay = 5e-4,
        scheduler    = "multistep",
        milestones   = [40, 80, 105],
        gamma        = 0.2,
        epochs       = 120,
        batch_size   = 256,
        mixup_alpha  = 0.4,
        label_smooth = 0.1,
        n            = 3,        # блоков на стадию: 2 → ~2.8M, 3 → ~4.5M, 4 → ~6.2M
        drop_rate    = 0.3,
        description  = "CifarSEResNet (PreActResNet + SE), ~4.5M params, 76.1% CIFAR-100, FL-friendly",
    ),
    "simple_cnn": dict(
        optimizer    = "sgd",
        lr           = 0.01,
        momentum     = 0.9,
        weight_decay = 1e-4,
        scheduler    = "cosine",
        epochs       = 60,
        batch_size   = 64,
        description  = "SimpleCNN, ~210K params, быстрый тест на CPU",
    ),
    "cct_7_3x1": dict(
        optimizer    = "adamw",
        lr           = 5e-4,
        weight_decay = 3e-2,   # трансформеры требуют сильнее wd
        scheduler    = "cosine",
        epochs       = 200,
        batch_size   = 128,
        mixup_alpha  = 0.4,
        label_smooth = 0.1,
        description  = "CCT-7/3×1 (Conv+Transformer), ~3.7M params",
    ),
}


def build_model(name: str, num_classes: int = 100, **kwargs) -> "torch.nn.Module":
    """Создать модель по имени.

    Args:
        name:        "wrn_28_4" | "wrn_28_10" | "efficientnet_b0" | "cct_7_3x1"
        num_classes: число классов (default 100)
        **kwargs:    переопределить дефолтные параметры модели
    """
    if name == "wrn_28_4":
        return WideResNet(depth=28, widen=4, num_classes=num_classes,
                          drop_rate=kwargs.get("drop_rate", 0.3))
    elif name == "wrn_28_10":
        return WideResNet(depth=28, widen=10, num_classes=num_classes,
                          drop_rate=kwargs.get("drop_rate", 0.3))
    elif name == "se_resnet":
        cfg = MODEL_CONFIGS["se_resnet"]
        return CifarSEResNet(
            num_classes=num_classes,
            n=kwargs.get("n", cfg["n"]),
            drop_rate=kwargs.get("drop_rate", cfg["drop_rate"]),
        )
    elif name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            drop_rate=kwargs.get("drop_rate", 0.2),
            drop_path_rate=kwargs.get("drop_path_rate", 0.2),
        )
    elif name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes,
                         drop_rate=kwargs.get("drop_rate", 0.25))
    elif name == "cct_7_3x1":
        return CCT(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_CONFIGS.keys()) + ['wrn_28_10']}"
        )
