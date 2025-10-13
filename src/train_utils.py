from __future__ import annotations
from typing import Dict
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def train_one_epoch(model: nn.Module, loader: DataLoader, device, optimizer, criterion) -> Dict[str, float]:
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"loss": total_loss/total, "acc": correct/total}

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"val_loss": total_loss/total, "val_acc": correct/total}