# food101_effnet/models.py
import timm
import torch.nn as nn

from .config import TrainConfig

def build_model(cfg: TrainConfig) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        # сюда потом можно добавить drop_rate / drop_path_rate
    )
    return model