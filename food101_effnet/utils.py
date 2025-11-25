# food101_effnet/utils.py
import json
import platform
from dataclasses import asdict
from pathlib import Path
import time

import torch
import torch.nn as nn

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred == targets).sum().item()
        return correct / targets.size(0)

def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, best_acc: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }, str(path))

def pick_device(device_flag: str) -> torch.device:
    if device_flag == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_env_and_device(run_path: Path, device: torch.device, amp_enabled: bool):
    import timm
    lines = []
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"PyTorch: {torch.__version__}")
    try:
        import torchvision
        lines.append(f"torchvision: {torchvision.__version__}")
    except Exception:
        lines.append("torchvision: n/a")
    lines.append(f"timm: {timm.__version__}")
    lines.append(f"Device: {device.type}")

    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            cc   = torch.cuda.get_device_capability(0)
            total, free = torch.cuda.mem_get_info()
            lines.append(f"GPU: {name} (CC {cc[0]}.{cc[1]})")
            lines.append(f"CUDA (torch): {torch.version.cuda}")
            lines.append(f"cuDNN: {torch.backends.cudnn.version()}")
            lines.append(f"Memory total: {total/1024**3:.2f} GB | free: {free/1024**3:.2f} GB")
        except Exception as e:
            lines.append(f"GPU info error: {e!r}")
    else:
        try:
            import psutil
            lines.append(f"CPU: {platform.processor()} | RAM: {psutil.virtual_memory().total/1024**3:.2f} GB")
        except Exception:
            lines.append(f"CPU: {platform.processor()}")

    lines.append(f"AMP enabled: {amp_enabled}")

    msg = "\n".join(lines)
    print("\n=== Environment ===\n" + msg + "\n===================\n")
    (run_path / "env.txt").write_text(msg, encoding="utf-8")