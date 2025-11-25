# food101_effnet/optim.py
import torch
from .config import TrainConfig

def make_optimizer(cfg: TrainConfig, params):
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )