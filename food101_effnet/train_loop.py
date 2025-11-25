# food101_effnet/train_loop.py
import time
import torch
from torch import amp as torch_amp

from .utils import accuracy_top1

def train_one_epoch(model, loader, device, criterion, optimizer, scaler, use_amp: bool):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    t0 = time.time()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch_amp.autocast(device_type=("cuda" if images.is_cuda else "cpu"), enabled=True):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_top1(logits, targets) * bs
        total_n    += bs
    epoch_time = time.time() - t0
    return total_loss / total_n, total_acc / total_n, epoch_time

@torch.no_grad()
def validate(model, loader, device, criterion, use_amp: bool):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    t0 = time.time()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_amp:
            with torch_amp.autocast(device_type=("cuda" if images.is_cuda else "cpu"), enabled=True):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_top1(logits, targets) * bs
        total_n    += bs
    epoch_time = time.time() - t0
    return total_loss / total_n, total_acc / total_n, epoch_time