from __future__ import annotations

import csv
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from torchvision.transforms import (
    Compose, Normalize, ToTensor,
    RandomHorizontalFlip, RandomCrop,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===============================
# CONFIG — меняешь только это
# ===============================

DATASET = "mnist"      # "cifar10" или "mnist"
DATA_DIR = "data"
OUT_DIR = "local_exp"

EPOCHS = 100
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-4
NESTEROV = True

SCHEDULER = "cosine"     # пока поддерживаем только cosine
COSINE_ETA_MIN = 0.0
WARMUP_EPOCHS = 5

LABEL_SMOOTHING = 0.1    # например 0.1 для CIFAR (по желанию)

NUM_WORKERS = 0
SEED = 42
FORCE_CPU = False


# ===============================
# Helpers (не крутилки)
# ===============================

def safe_tag(s: str) -> str:
    s = str(s).strip().replace("/", "-").replace("\\", "-").replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "exp"


def plot_curves(train_vals, val_vals, title: str, ylabel: str, out_path: Path) -> None:
    epochs = list(range(1, len(train_vals) + 1))
    plt.figure()
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def model_summary(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "repr": repr(model),
    }


def infer_num_classes(ds, label_col: str, limit: int = 10_000) -> int:
    mx = 0
    for i in range(min(limit, len(ds))):
        y = int(ds[i][label_col])
        if y > mx:
            mx = y
    return mx + 1


def make_criterion() -> nn.Module:
    if LABEL_SMOOTHING and LABEL_SMOOTHING > 0:
        return nn.CrossEntropyLoss(label_smoothing=float(LABEL_SMOOTHING))
    return nn.CrossEntropyLoss()


# ===============================
# Dataset-specific settings (фиксировано)
# ===============================

def dataset_config(name: str) -> Dict[str, Any]:
    name = name.lower()

    if "cifar" in name:
        return {
            "img_col": "img",
            "label_col": "label",
            "in_channels": 3,
            "img_size": 32,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
            "train_padding": 4,
            "hflip": True,
        }

    if "mnist" in name:
        return {
            "img_col": "image",
            "label_col": "label",
            "in_channels": 1,
            "img_size": 28,
            "mean": (0.1307,),
            "std": (0.3081,),
            "train_padding": 2,
            "hflip": False,
        }

    raise ValueError("Dataset must be 'cifar10' or 'mnist'")


# ===============================
# Model (архитектура как была)
# ===============================

class ImprovedCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ===============================
# Train / Eval
# ===============================

def train_one_epoch(model, loader, device, optimizer, img_col, label_col) -> tuple[float, float]:
    model.train()
    criterion = make_criterion()

    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch[img_col].to(device)
        y = batch[label_col].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, img_col, label_col) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x = batch[img_col].to(device)
        y = batch[label_col].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ===============================
# Main
# ===============================

def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    cfg = dataset_config(DATASET)
    img_col = cfg["img_col"]
    label_col = cfg["label_col"]

    # ---- unique experiment dir ----
    created_at = datetime.now().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = safe_tag(
        f"{DATASET}__bs{BATCH_SIZE}__lr{LR}__wd{WEIGHT_DECAY}__seed{SEED}__{run_id}"
    )
    out_dir = Path(OUT_DIR) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load dataset ----
    ds_path = Path(DATA_DIR) / DATASET
    ds = load_from_disk(str(ds_path))

    raw_train_ds = ds["train"]
    raw_val_ds = ds["test"] if "test" in ds else ds["validation"]

    # ---- infer classes on RAW dataset (без transforms) ----
    num_classes = infer_num_classes(raw_train_ds, label_col)
    print(f"Classes: {num_classes}")

    # ---- transforms (фиксировано под CIFAR/MNIST) ----
    train_tfms = [
        RandomCrop(cfg["img_size"], padding=cfg["train_padding"]),
    ]
    if cfg["hflip"]:
        train_tfms.append(RandomHorizontalFlip(p=0.5))

    train_tfms += [
        ToTensor(),
        Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]

    train_transform = Compose(train_tfms)
    val_transform = Compose([
        ToTensor(),
        Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    def apply_train_tfm(batch):
        batch[img_col] = [train_transform(img) for img in batch[img_col]]
        return batch

    def apply_val_tfm(batch):
        batch[img_col] = [val_transform(img) for img in batch[img_col]]
        return batch

    train_ds = raw_train_ds.with_transform(apply_train_tfm)
    val_ds = raw_val_ds.with_transform(apply_val_tfm)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    # ---- model ----
    model = ImprovedCNN(in_channels=cfg["in_channels"], num_classes=num_classes).to(device)
    ms = model_summary(model)
    print(f"Trainable params: {ms['trainable_params']:,}")

    # ---- optimizer ----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(LR),
        momentum=float(MOMENTUM),
        weight_decay=float(WEIGHT_DECAY),
        nesterov=bool(NESTEROV),
    )

    # ---- scheduler ----
    if SCHEDULER != "cosine":
        raise ValueError(f"Only SCHEDULER='cosine' supported now, got '{SCHEDULER}'")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=int(WARMUP_EPOCHS)
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(EPOCHS - WARMUP_EPOCHS), eta_min=float(COSINE_ETA_MIN)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(WARMUP_EPOCHS)],
    )

    # ---- paths ----
    metrics_path = out_dir / "metrics.csv"
    best_path = out_dir / "best_model.pt"
    cfg_path = out_dir / "config.json"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "sec"])

    best_val_acc = 0.0
    best_epoch = 0

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # ---- train loop ----
    for epoch in range(1, int(EPOCHS) + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, img_col, label_col)
        va_loss, va_acc = evaluate(model, val_loader, device, img_col, label_col)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        dt = time.perf_counter() - t0

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            marker = " ← best"

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {va_loss:.4f}/{va_acc:.4f} | "
            f"lr {current_lr:.5f} | {dt:.1f}s{marker}"
        )

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, current_lr, dt])

    # ---- plots ----
    plot_curves(train_losses, val_losses, title=f"{exp_name} - Loss", ylabel="loss", out_path=out_dir / "loss.png")
    plot_curves(train_accs, val_accs, title=f"{exp_name} - Accuracy", ylabel="accuracy", out_path=out_dir / "acc.png")

    # ---- config.json: только параметры обучения из CONFIG + результаты ----
    config: Dict[str, Any] = {
        "created_at": created_at,
        "exp_name": exp_name,
        "paths": {
            "out_dir": str(out_dir),
            "dataset_path": str(ds_path),
            "metrics_csv": str(metrics_path),
            "best_model_pt": str(best_path),
            "loss_png": str(out_dir / "loss.png"),
            "acc_png": str(out_dir / "acc.png"),
        },
        "train_config": {
            "DATASET": DATASET,
            "DATA_DIR": DATA_DIR,
            "OUT_DIR": OUT_DIR,
            "EPOCHS": int(EPOCHS),
            "BATCH_SIZE": int(BATCH_SIZE),
            "LR": float(LR),
            "MOMENTUM": float(MOMENTUM),
            "WEIGHT_DECAY": float(WEIGHT_DECAY),
            "NESTEROV": bool(NESTEROV),
            "SCHEDULER": SCHEDULER,
            "COSINE_ETA_MIN": float(COSINE_ETA_MIN),
            "WARMUP_EPOCHS": int(WARMUP_EPOCHS),
            "LABEL_SMOOTHING": float(LABEL_SMOOTHING),
            "NUM_WORKERS": int(NUM_WORKERS),
            "SEED": int(SEED),
            "FORCE_CPU": bool(FORCE_CPU),
        },
        "results": {
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "num_classes": int(num_classes),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
        },
        "model_summary": ms,
    }

    cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"✅ Saved best:   {best_path.resolve()}")
    print(f"✅ Saved metrics:{metrics_path.resolve()}")
    print(f"✅ Saved plots:  {(out_dir / 'loss.png').resolve()} , {(out_dir / 'acc.png').resolve()}")
    print(f"✅ Saved config: {cfg_path.resolve()}")


if __name__ == "__main__":
    main()