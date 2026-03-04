from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _infer_columns(ds) -> Tuple[str, str]:
    """Определить имена колонок изображения и метки в HuggingFace Dataset."""
    keys = set(ds.features.keys())
    img_col = next((c for c in ("img", "image", "pixel_values") if c in keys), None)
    if img_col is None:
        raise KeyError(f"Колонка с изображением не найдена. Доступные: {sorted(keys)}")
    label_col = "label" if "label" in keys else "labels"
    return img_col, label_col


def make_dataloader(
    partition_path: Path | str,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
) -> Tuple[DataLoader, str, str]:
    """Загрузить партицию с диска и вернуть DataLoader + имена колонок."""
    from torchvision.transforms import ToTensor

    ds = load_from_disk(str(partition_path))
    img_col, label_col = _infer_columns(ds)

    def to_tensor(batch):
        batch[img_col] = [ToTensor()(x) for x in batch[img_col]]
        return batch

    ds = ds.with_transform(to_tensor)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, img_col, label_col


def local_train(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    img_col: str,
    label_col: str,
    mu: float = 0.0,
) -> Tuple[List[float], int]:
    """Локальное обучение клиента.

    Args:
        mu: коэффициент FedProx proximal term (0.0 = выключен).

    Returns:
        (epoch_losses, num_examples)
    """
    model.to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    global_params = (
        [p.data.clone().to(device) for p in model.parameters()] if mu > 0.0 else None
    )

    epoch_losses: List[float] = []
    for _ in range(epochs):
        loss_sum, batches = 0.0, 0
        for batch in loader:
            x = batch[img_col].to(device)
            y = batch[label_col].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if global_params is not None:
                prox = sum(
                    (w - wg).pow(2).sum()
                    for w, wg in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2) * prox
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        epoch_losses.append(loss_sum / max(batches, 1))

    return epoch_losses, len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    img_col: str,
    label_col: str,
) -> Tuple[float, float, Dict[int, Tuple[int, int]]]:
    """Серверная оценка модели.

    Returns:
        (loss, accuracy, per_class) где per_class = {class_id: (correct, total)}
    """
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, batches, correct, total = 0.0, 0, 0, 0
    per_class: Dict[int, List[int]] = {}  # {class_id: [correct, total]}

    for batch in loader:
        x      = batch[img_col].to(device)
        y      = batch[label_col].to(device)
        logits = model(x)

        total_loss += float(criterion(logits, y).item())
        batches    += 1

        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum())
        total   += y.numel()

        for true_cls, pred_cls in zip(y.tolist(), preds.tolist()):
            if true_cls not in per_class:
                per_class[true_cls] = [0, 0]
            per_class[true_cls][1] += 1
            if true_cls == pred_cls:
                per_class[true_cls][0] += 1

    per_class_result: Dict[int, Tuple[int, int]] = {
        cls: (counts[0], counts[1]) for cls, counts in per_class.items()
    }
    return total_loss / max(batches, 1), correct / max(total, 1), per_class_result
