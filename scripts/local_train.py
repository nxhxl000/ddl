# local_train.py
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict, load_from_disk

import matplotlib.pyplot as plt

try:
    import torchvision
    from torchvision.transforms import ToTensor, Normalize, Compose
    from torchvision import models as tv_models
except Exception as e:
    raise RuntimeError("torchvision is required. Install torchvision and try again.") from e


# ============================================================
# CONFIG (EDIT HERE)
# ============================================================

DATA_DIR = "data"          # где лежат датасеты (data/<dataset>)
MODELS_DIR = "models"      # где лежат модели (models/<kind>/<model>.pt)
OUT_ROOT = "local_exp"     # куда сохранять результаты

# Выбираешь, что обучаем:
DATASET = "cifar10"        # имя папки в data/, либо абсолютный/относительный путь к датасету
MODEL_ID = "resnet18_cifar"  # cifar: simple_cnn | resnet18_cifar | mobilenet_v3_small
                             # mnist: simple_cnn | lenet5 | mobilenet_v3_small_1ch

# Какой split считать валидацией:
VAL_SPLIT_PRIORITY = ("test", "validation")  # сначала ищем test, если нет — validation

# Гиперпараметры обучения:
EPOCHS = 20
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
SEED = 42

# Нормализация (по умолчанию для MNIST/CIFAR)
USE_NORMALIZATION = True

# Если хочешь принудительно CPU:
FORCE_CPU = False

# ============================================================


# -------------------------
# Dataset utilities
# -------------------------

def infer_label_name(ds: Dataset) -> str:
    """Определяет имя колонки с метками (label/labels) в датасете HuggingFace."""
    keys = set(ds.features.keys())
    if "label" in keys:
        return "label"
    if "labels" in keys:
        return "labels"
    raise KeyError(f"Label column not found. Available columns: {sorted(keys)}")


def infer_image_name(ds: Dataset) -> str:
    """Определяет имя колонки с изображением (img/image/pixel_values) в датасете HuggingFace."""
    keys = set(ds.features.keys())
    for k in ("img", "image", "pixel_values"):
        if k in keys:
            return k
    raise KeyError(f"Image column not found. Available columns: {sorted(keys)}")


def infer_num_classes(ds: Dataset, label_col: str) -> int:
    """Определяет количество классов: через ClassLabel, иначе — проход по части датасета."""
    feat = ds.features.get(label_col)
    if hasattr(feat, "names") and feat.names is not None:  # HF ClassLabel
        return int(len(feat.names))
    mx = 0
    for i in range(min(10_000, len(ds))):
        y = int(ds[i][label_col])
        if y > mx:
            mx = y
    return int(mx + 1)


def guess_dataset_kind(name_or_path: str) -> str:
    """Грубо определяет тип датасета: mnist/cifar по имени папки/пути (иначе — cifar)."""
    s = str(name_or_path).lower()
    if "mnist" in s:
        return "mnist"
    if "cifar" in s:
        return "cifar"
    return "cifar"


def safe_tag(s: str) -> str:
    """Преобразует строку в безопасный тег для имени папки/файла."""
    s = str(s).strip().replace("/", "-").replace("\\", "-").replace(" ", "-")
    import re
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "unknown"


def get_default_mean_std(dataset_kind: str) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """
    Возвращает стандартные mean/std для нормализации популярных датасетов.
    Если датасет неизвестен — None (тогда будет только ToTensor()).
    """
    dk = dataset_kind.lower()

    # MNIST (1 канал)
    if dk == "mnist":
        return (0.1307,), (0.3081,)

    # CIFAR-10/100 (3 канала) — общепринятые статистики CIFAR-10
    if dk == "cifar":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    return None


def make_image_transform(dataset_kind: str, use_normalization: bool = True):
    """
    Собирает torchvision transform:
    - всегда ToTensor()
    - + Normalize(mean,std), если включено и есть дефолтные статистики.
    """
    if not use_normalization:
        return ToTensor()

    ms = get_default_mean_std(dataset_kind)
    if ms is None:
        return ToTensor()

    mean, std = ms
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])


# -------------------------
# Local loading
# -------------------------

def load_local_dataset(dataset: str, data_dir: str) -> DatasetDict:
    """
    Загружает датасет ТОЛЬКО локально.
    Ищет папку:
      - dataset (если это путь)
      - data_dir/dataset
    Возвращает DatasetDict (train/test/validation и т.п.).
    """
    p = Path(dataset)
    if p.exists() and p.is_dir():
        obj = load_from_disk(str(p))
    else:
        p2 = Path(data_dir) / dataset
        if not (p2.exists() and p2.is_dir()):
            raise FileNotFoundError(
                f"Local dataset not found. Tried: '{p}' and '{p2}'. "
                "Place dataset in data/<dataset> or set DATASET to a path."
            )
        obj = load_from_disk(str(p2))

    if isinstance(obj, DatasetDict):
        return obj
    return DatasetDict({"train": obj})


def pick_val_split(ds: DatasetDict, priority: tuple[str, ...]) -> Dataset:
    """Выбирает split для валидации по приоритету (например, test -> validation)."""
    for name in priority:
        if name in ds:
            return ds[name]
    raise KeyError(f"No validation split found. Tried {priority}. Available: {list(ds.keys())}")


def load_local_model_bundle(models_dir: str, dataset_kind: str, model_id: str) -> Tuple[Dict[str, Any], Path]:
    """
    Загружает bundle модели из локальной папки models/<kind>/<model_id>.pt.
    Ожидает формат: {"state_dict": ..., "meta": ...}.
    """
    model_path = Path(models_dir) / dataset_kind / f"{model_id}.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Expected: {models_dir}/{dataset_kind}/{model_id}.pt"
        )
    bundle = torch.load(str(model_path), map_location="cpu")
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError(f"Invalid model bundle in {model_path} (expected dict with 'state_dict').")
    return bundle, model_path


# -------------------------
# Model builders
# -------------------------

def build_simple_cnn(in_channels: int, num_classes: int) -> nn.Module:
    """Строит простой CNN (универсальный для MNIST/CIFAR)."""
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.gap(x).flatten(1)
            return self.fc(x)

    return Net()


def build_lenet5(num_classes: int) -> nn.Module:
    """Строит LeNet-5 (классическая CNN для MNIST)."""
    class LeNet5(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 28->24
            self.pool = nn.AvgPool2d(2, 2)                # 24->12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 12->8
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.tanh(self.conv1(x))
            x = self.pool(x)
            x = torch.tanh(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return self.fc3(x)

    return LeNet5()


def build_resnet18_cifar(num_classes: int) -> nn.Module:
    """Строит ResNet18, адаптированный под CIFAR (conv1=3x3 stride1, без maxpool)."""
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _replace_first_conv_mobilenet_v3(model: nn.Module, in_channels: int) -> None:
    """Заменяет первый Conv2d слой у MobileNetV3, чтобы поддержать 1-канальный вход (MNIST)."""
    first = model.features[0][0]
    if not isinstance(first, nn.Conv2d):
        raise RuntimeError("Unexpected MobileNetV3 structure: first layer is not Conv2d")
    model.features[0][0] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        dilation=first.dilation,
        groups=first.groups,
        bias=(first.bias is not None),
        padding_mode=first.padding_mode,
    )


def build_mobilenet_v3_small(num_classes: int, in_channels: int) -> nn.Module:
    """Строит MobileNetV3-Small и настраивает входные каналы + последний классификатор."""
    model = tv_models.mobilenet_v3_small(weights=None)
    _replace_first_conv_mobilenet_v3(model, in_channels=in_channels)
    last = model.classifier[-1]
    if not isinstance(last, nn.Linear):
        raise RuntimeError("Unexpected classifier structure in MobileNetV3")
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def build_model_for_dataset(model_id: str, dataset_kind: str, num_classes: int) -> nn.Module:
    """
    Возвращает архитектуру модели по строковому идентификатору.
    dataset_kind влияет на число каналов входа и допустимые model_id.
    """
    dataset_kind = dataset_kind.lower()

    if dataset_kind == "cifar":
        if model_id == "simple_cnn":
            return build_simple_cnn(in_channels=3, num_classes=num_classes)
        if model_id == "resnet18_cifar":
            return build_resnet18_cifar(num_classes=num_classes)
        if model_id == "mobilenet_v3_small":
            return build_mobilenet_v3_small(num_classes=num_classes, in_channels=3)
        raise ValueError(f"Unknown CIFAR model_id='{model_id}'")

    if dataset_kind == "mnist":
        if model_id == "simple_cnn":
            return build_simple_cnn(in_channels=1, num_classes=num_classes)
        if model_id == "lenet5":
            return build_lenet5(num_classes=num_classes)
        if model_id == "mobilenet_v3_small_1ch":
            return build_mobilenet_v3_small(num_classes=num_classes, in_channels=1)
        raise ValueError(f"Unknown MNIST model_id='{model_id}'")

    raise ValueError(f"Unknown dataset_kind='{dataset_kind}' (expected 'cifar' or 'mnist')")


# -------------------------
# Training/eval
# -------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, img_col: str, label_col: str) -> Tuple[float, float]:
    """Считает loss и accuracy на валидационном/тестовом DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0

    for batch in loader:
        x = batch[img_col].to(device)
        y = batch[label_col].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        total_batches += 1

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return total_loss / max(total_batches, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_col: str,
    label_col: str,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Обучает модель одну эпоху и возвращает (train_loss, train_acc)."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0

    for batch in loader:
        x = batch[img_col].to(device)
        y = batch[label_col].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return total_loss / max(total_batches, 1), correct / max(total, 1)


# -------------------------
# Plotting / summaries
# -------------------------

def plot_curves(values_train, values_val, title: str, ylabel: str, out_path: Path) -> None:
    """Строит и сохраняет график для train/val кривых (loss или accuracy)."""
    epochs = list(range(1, len(values_train) + 1))
    plt.figure()
    plt.plot(epochs, values_train, label="train")
    plt.plot(epochs, values_val, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def model_summary(model: nn.Module) -> Dict[str, Any]:
    """Считает простую сводку по модели (число параметров + repr)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "repr": repr(model),
    }


# -------------------------
# Main
# -------------------------

def main() -> None:
    """Точка входа: загружает локальный датасет+модель, обучает, логирует, рисует графики."""
    torch.manual_seed(SEED)

    dataset_kind = guess_dataset_kind(DATASET)

    # ---- device ----
    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- load local dataset ----
    ds_dict = load_local_dataset(DATASET, DATA_DIR)
    if "train" not in ds_dict:
        raise KeyError(f"Local dataset has no 'train' split. Available: {list(ds_dict.keys())}")

    train_ds = ds_dict["train"]
    val_ds = pick_val_split(ds_dict, VAL_SPLIT_PRIORITY)

    img_col = infer_image_name(train_ds)
    label_col = infer_label_name(train_ds)
    num_classes = infer_num_classes(train_ds, label_col)

    # ---- transforms (ToTensor + optional Normalize) ----
    transform = make_image_transform(dataset_kind, use_normalization=USE_NORMALIZATION)

    def tfm(batch):
        batch[img_col] = [transform(x) for x in batch[img_col]]
        return batch

    train_ds = train_ds.with_transform(tfm)
    val_ds = val_ds.with_transform(tfm)

    pin_memory = torch.cuda.is_available() and (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # ---- load model bundle ----
    bundle, model_path = load_local_model_bundle(MODELS_DIR, dataset_kind, MODEL_ID)

    # ---- build model + load weights ----
    model = build_model_for_dataset(model_id=MODEL_ID, dataset_kind=dataset_kind, num_classes=num_classes)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(LR),
        momentum=float(MOMENTUM),
        weight_decay=float(WEIGHT_DECAY),
    )

    # ---- experiment dir ----
    dataset_tag = safe_tag(Path(DATASET).name if Path(DATASET).exists() else DATASET)
    exp_name = f"{dataset_tag}__{safe_tag(MODEL_ID)}"
    exp_dir = Path(OUT_ROOT) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- config payload ----
    created_at = datetime.now().isoformat(timespec="seconds")
    cfg: Dict[str, Any] = {
        "created_at": created_at,
        "experiment": exp_name,
        "config": {
            "DATASET": DATASET,
            "DATA_DIR": DATA_DIR,
            "dataset_kind": dataset_kind,
            "VAL_SPLIT_PRIORITY": list(VAL_SPLIT_PRIORITY),
            "MODEL_ID": MODEL_ID,
            "MODELS_DIR": MODELS_DIR,
            "EPOCHS": int(EPOCHS),
            "BATCH_SIZE": int(BATCH_SIZE),
            "LR": float(LR),
            "MOMENTUM": float(MOMENTUM),
            "WEIGHT_DECAY": float(WEIGHT_DECAY),
            "NUM_WORKERS": int(NUM_WORKERS),
            "SEED": int(SEED),
            "FORCE_CPU": bool(FORCE_CPU),
            "USE_NORMALIZATION": bool(USE_NORMALIZATION),
            "device": str(device),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": getattr(torchvision, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "dataset": {
            "splits": list(ds_dict.keys()),
            "train_size": int(len(train_ds)),
            "val_size": int(len(val_ds)),
            "img_col": img_col,
            "label_col": label_col,
            "num_classes": int(num_classes),
        },
        "model_bundle": {
            "path": str(model_path),
            "meta": bundle.get("meta", {}),
        },
        "model_summary": model_summary(model),
    }
    (exp_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- log files ----
    log_file = exp_dir / "train.log"
    metrics_csv = exp_dir / "metrics.csv"

    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Start: {created_at}\n")
        f.write(f"Dataset kind: {dataset_kind}\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Model file: {model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Normalization: {USE_NORMALIZATION} (default stats for {dataset_kind})\n")
        f.write(
            f"EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LR}, MOMENTUM={MOMENTUM}, "
            f"WEIGHT_DECAY={WEIGHT_DECAY}, NUM_WORKERS={NUM_WORKERS}, SEED={SEED}\n\n"
        )

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"])

    # ---- train loop ----
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, int(EPOCHS) + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, img_col, label_col, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, device, img_col, label_col)

        dt = time.perf_counter() - t0

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        with log_file.open("a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch:03d}: "
                f"train_loss={tr_loss:.6f}, train_acc={tr_acc:.4f} | "
                f"val_loss={va_loss:.6f}, val_acc={va_acc:.4f} | "
                f"time={dt:.2f}s\n"
            )

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc, dt])

    # ---- plots ----
    plot_curves(train_losses, val_losses, title=f"{exp_name} - Loss", ylabel="loss", out_path=exp_dir / "loss.png")
    plot_curves(train_accs, val_accs, title=f"{exp_name} - Accuracy", ylabel="accuracy", out_path=exp_dir / "acc.png")

    # ---- save final model ----
    torch.save(model.state_dict(), exp_dir / "final_model.pt")

    with log_file.open("a", encoding="utf-8") as f:
        f.write("\nDone.\n")
        f.write(f"Saved config: {exp_dir / 'config.json'}\n")
        f.write(f"Saved metrics: {exp_dir / 'metrics.csv'}\n")
        f.write(f"Saved plots: {exp_dir / 'loss.png'}, {exp_dir / 'acc.png'}\n")
        f.write(f"Saved model: {exp_dir / 'final_model.pt'}\n")

    print("✅ Done.")
    print("Experiment dir:", exp_dir.resolve())


if __name__ == "__main__":
    main()
