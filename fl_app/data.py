"""Загрузка партиций + трансформы + DataLoader (CIFAR-100 / PlantVillage)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Nomralization stats + image column per dataset
_DATASET_INFO: dict[str, dict] = {
    "cifar100": {
        "mean":      (0.5071, 0.4866, 0.4409),
        "std":       (0.2673, 0.2564, 0.2762),
        "img_col":   "img",
        "label_col": "fine_label",
        "size":      32,
    },
    "plantvillage": {
        "mean":      (0.485, 0.456, 0.406),
        "std":       (0.229, 0.224, 0.225),
        "img_col":   "image",
        "label_col": "label",
        "size":      224,
    },
}


def _train_transform(dataset: str):
    info = _DATASET_INFO[dataset]
    if dataset == "cifar100":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
    if dataset == "plantvillage":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
    raise ValueError(f"Unknown dataset: {dataset}")


def _eval_transform(dataset: str):
    info = _DATASET_INFO[dataset]
    if dataset == "cifar100":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
    if dataset == "plantvillage":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
    raise ValueError(f"Unknown dataset: {dataset}")


class _HFDataset(Dataset):
    """HuggingFace Dataset → torch Dataset с transform."""

    def __init__(self, hf_ds, tf, img_col: str, label_col: str):
        self.ds = hf_ds
        self.tf = tf
        self.img_col = img_col
        self.label_col = label_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item[self.img_col]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.tf(img), item[self.label_col]


def _stratified_indices(labels, fraction: float, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    rng = random.Random(seed)
    picked: list[int] = []
    for y, idxs in by_class.items():
        k = max(1, int(round(len(idxs) * fraction)))
        rng.shuffle(idxs)
        picked.extend(idxs[:k])
    picked.sort()
    return picked


def _resolve_dataset(partition_dir: Path) -> str:
    """Определить dataset по manifest.json в родительской директории партиции."""
    manifest = partition_dir.parent / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["dataset"]
    # Фолбэк для старых партиций CIFAR-100 без манифеста
    return "cifar100"


def build_loader(
    partition_dir: str | Path,
    *,
    batch_size: int,
    train: bool,
    chunk_fraction: float = 1.0,
    chunk_seed: int = 0,
) -> DataLoader:
    """Прочитать партицию с диска и построить DataLoader.

    Dataset определяется по manifest.json в родительской директории партиции.
    При chunk_fraction<1.0 делается стратифицированный сабсэмпл.
    """
    partition_dir = Path(partition_dir)
    dataset = _resolve_dataset(partition_dir)
    info = _DATASET_INFO[dataset]
    img_col, label_col = info["img_col"], info["label_col"]

    hf_ds = load_from_disk(str(partition_dir))
    if train and chunk_fraction < 1.0:
        idx = _stratified_indices(hf_ds[label_col], chunk_fraction, chunk_seed)
        hf_ds = hf_ds.select(idx)
    tf = _train_transform(dataset) if train else _eval_transform(dataset)
    return DataLoader(
        _HFDataset(hf_ds, tf, img_col, label_col),
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
