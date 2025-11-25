# food101_effnet/data.py
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment

from .config import TrainConfig

MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)

def make_transforms(cfg: TrainConfig):
    tf_val = transforms.Compose([
        transforms.Resize(cfg.resize_short),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    if not cfg.aug_train:
        return tf_val, tf_val

    preset = cfg.aug_preset.lower()

    if preset == "basic":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif preset == "geo":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif preset == "color":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif preset == "erasing":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25),
        ])
    elif preset == "auto":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif preset == "rand":
        tf_train = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        tf_train = tf_val

    return tf_train, tf_val

def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, list]:
    tf_train, tf_val = make_transforms(cfg)
    train_set = datasets.Food101(root=cfg.data_dir, split="train", download=False, transform=tf_train)
    val_set   = datasets.Food101(root=cfg.data_dir, split="test",  download=False, transform=tf_val)
    classes = getattr(train_set, "classes", list(range(cfg.num_classes)))

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0
    )
    return train_loader, val_loader, classes