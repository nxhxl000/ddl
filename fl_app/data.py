"""Загрузка CIFAR-100 партиций + трансформы + DataLoader."""

from __future__ import annotations

from pathlib import Path

from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def _train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


class _HFDataset(Dataset):
    """HuggingFace Dataset → torch Dataset с transform."""

    def __init__(self, hf_ds, tf):
        self.ds, self.tf = hf_ds, tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return self.tf(item["img"]), item["fine_label"]


def build_loader(partition_dir: str | Path, *, batch_size: int, train: bool) -> DataLoader:
    """Прочитать партицию с диска и построить DataLoader.

    Ожидается HuggingFace Dataset с колонками 'img' (PIL) и 'fine_label' (int).
    """
    hf_ds = load_from_disk(str(partition_dir))
    tf = _train_transform() if train else _eval_transform()
    return DataLoader(
        _HFDataset(hf_ds, tf),
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
