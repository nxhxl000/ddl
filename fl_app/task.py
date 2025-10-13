from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import build_model

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def build_server_eval_loader(data_dir: str = "data") -> DataLoader:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    testset = datasets.CIFAR10(data_dir, train=False, download=False, transform=tf)
    pin = torch.cuda.is_available()
    return DataLoader(testset, batch_size=512, shuffle=False, num_workers=0, pin_memory=pin)

@torch.no_grad()
def server_evaluate_fn(loader: DataLoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    def evaluate(server_round: int, parameters, config):
        model = build_model().to(device)
        sd_keys = list(model.state_dict().keys())
        assert len(sd_keys) == len(parameters), "Mismatch in parameter count"

        new_sd = {k: torch.as_tensor(np.array(v), device=device) for k, v in zip(sd_keys, parameters)}
        model.load_state_dict(new_sd, strict=True)

        total, correct, total_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return float(total_loss / total), {"val_acc": float(correct / total)}

    return evaluate