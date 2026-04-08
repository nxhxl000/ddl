from __future__ import annotations

import random
from math import floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def stratified_select(ds, budget: int, label_col: str, seed: int):
    """Выбрать `budget` сэмплов из ds с сохранением распределения классов.

    Каждый класс получает floor/ceil(budget * class_fraction) сэмплов,
    выбранных с детерминированным перемешиванием (меняется по раундам).
    Это предотвращает деградацию редких классов при чанкинге.
    """
    labels = ds[label_col]
    n_total = len(labels)
    budget = min(budget, n_total)

    # Группируем индексы по классам
    class_indices: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl not in class_indices:
            class_indices[lbl] = []
        class_indices[lbl].append(i)

    # Пропорциональная квота на каждый класс (floor + распределение остатка)
    quotas_f = {cls: budget * len(idxs) / n_total for cls, idxs in class_indices.items()}
    quotas   = {cls: floor(q) for cls, q in quotas_f.items()}
    remainder = budget - sum(quotas.values())
    # Остаток отдаём классам с наибольшей дробной частью
    for cls in sorted(class_indices, key=lambda c: -(quotas_f[c] - quotas[c]))[:remainder]:
        quotas[cls] += 1

    # Перемешиваем каждый класс и берём квоту
    rng = random.Random(seed)
    selected: List[int] = []
    for cls, idxs in class_indices.items():
        rng.shuffle(idxs)
        selected.extend(idxs[:quotas[cls]])

    # Финальное перемешивание чтобы классы не шли блоками
    rng.shuffle(selected)
    return ds.select(selected)


def _infer_columns(ds) -> Tuple[str, str]:
    """Определить имена колонок изображения и метки в HuggingFace Dataset."""
    keys = set(ds.features.keys())
    img_col = next((c for c in ("img", "image", "pixel_values") if c in keys), None)
    if img_col is None:
        raise KeyError(f"Колонка с изображением не найдена. Доступные: {sorted(keys)}")
    label_col = next(
        (c for c in ("label", "labels", "fine_label", "coarse_label") if c in keys),
        None,
    )
    if label_col is None:
        raise KeyError(f"Колонка с меткой не найдена. Доступные: {sorted(keys)}")
    return img_col, label_col


class _TensorDataset(torch.utils.data.Dataset):
    """Предобработанный датасет: PIL→Tensor один раз, потом из памяти."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.images = images    # (N, C, H, W) float32
        self.labels = labels    # (N,) long
        self.augment = augment
        if augment:
            from torchvision.transforms import RandomCrop, RandomHorizontalFlip
            self.crop = RandomCrop(images.shape[-1], padding=4)
            self.flip = RandomHorizontalFlip()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = self.crop(img)
            img = self.flip(img)
        return img, self.labels[idx]


def make_dataloader(
    partition_path: Path | str,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    augment: bool = False,
    dataset=None,
) -> Tuple[DataLoader, str, str]:
    """Загрузить партицию с диска и вернуть DataLoader + имена колонок.

    Конвертирует PIL→Tensor один раз при загрузке (кеш в памяти).
    Аугментация (RandomCrop, RandomHorizontalFlip) применяется на лету к тензорам.

    dataset: опционально передать уже загруженный HF Dataset.
             Если задан, partition_path игнорируется.
    """
    from torchvision.transforms.functional import to_tensor

    ds = dataset if dataset is not None else load_from_disk(str(partition_path))
    img_col, label_col = _infer_columns(ds)

    # Конвертируем все PIL→Tensor один раз
    images = torch.stack([to_tensor(x) for x in ds[img_col]])
    labels = torch.tensor(ds[label_col], dtype=torch.long)

    tensor_ds = _TensorDataset(images, labels, augment=augment)
    loader = DataLoader(
        tensor_ds,
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
    c_i: Optional[Dict[str, torch.Tensor]] = None,
    c_server: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[List[float], int, int]:
    """Локальное обучение клиента.

    Args:
        mu:       FedProx proximal term (0.0 = выключен).
        c_i:      SCAFFOLD client control variate {param_name: tensor}.
        c_server: SCAFFOLD server control variate {param_name: tensor}.

    Returns:
        (epoch_losses, num_examples, total_steps)
        total_steps нужен для вычисления c_i_new в SCAFFOLD.
    """
    model.to(device).train()
    # SCAFFOLD требует чистый SGD без momentum и без lr scheduler:
    # формула c_i_new = (x - y_i) / (K * lr) выведена для plain SGD с постоянным lr.
    # Momentum меняет эффективный шаг в (1/(1-β)) ≈ 10 раз → формула ломается → взрыв.
    scaffold_mode = c_i is not None and c_server is not None
    if scaffold_mode:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay
        )
        scheduler = None
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    global_params = (
        [p.data.clone().to(device) for p in model.parameters()] if mu > 0.0 else None
    )

    # SCAFFOLD: поправка к градиенту (c_server - c_i), переносим на device заранее
    scaffold_corr: Optional[Dict[str, torch.Tensor]] = None
    if scaffold_mode:
        scaffold_corr = {
            name: (c_server.get(name, torch.zeros_like(p.data))
                   - c_i.get(name, torch.zeros_like(p.data))).to(device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    epoch_losses: List[float] = []
    total_steps = 0
    for _ in range(epochs):
        loss_sum, batches = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if global_params is not None:
                prox = sum(
                    (w - wg).pow(2).sum()
                    for w, wg in zip(model.parameters(), global_params)
                )
                loss = loss + (mu / 2) * prox
            loss.backward()
            # Применяем SCAFFOLD поправку к градиентам
            if scaffold_corr is not None:
                for name, p in model.named_parameters():
                    if p.grad is not None and name in scaffold_corr:
                        p.grad.add_(scaffold_corr[name])
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
            total_steps += 1
        epoch_losses.append(loss_sum / max(batches, 1))
        if scheduler is not None:
            scheduler.step()

    return epoch_losses, len(loader.dataset), total_steps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    img_col: str,
    label_col: str,
) -> Tuple[float, float, Dict[int, Tuple[int, int]], float]:
    """Серверная оценка модели.

    Returns:
        (loss, accuracy, per_class, f1_macro)
        per_class = {class_id: (correct, total)}
        f1_macro  = macro-averaged F1 score across all classes
    """
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, batches, correct, total = 0.0, 0, 0, 0
    per_class: Dict[int, List[int]] = {}   # {class_id: [correct, total]}
    pred_counts: Dict[int, int] = {}       # {class_id: n_predicted} — needed for precision

    for x, y in loader:
        x      = x.to(device)
        y      = y.to(device)
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
            pred_counts[pred_cls] = pred_counts.get(pred_cls, 0) + 1

    # Macro F1: average per-class F1 over all true classes
    f1_scores = []
    for cls, (tp, n_true) in per_class.items():
        n_pred    = pred_counts.get(cls, 0)
        precision = tp / n_pred  if n_pred  > 0 else 0.0
        recall    = tp / n_true  if n_true  > 0 else 0.0
        denom     = precision + recall
        f1_scores.append(2 * precision * recall / denom if denom > 0 else 0.0)
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    per_class_result: Dict[int, Tuple[int, int]] = {
        cls: (counts[0], counts[1]) for cls, counts in per_class.items()
    }
    return total_loss / max(batches, 1), correct / max(total, 1), per_class_result, f1_macro
