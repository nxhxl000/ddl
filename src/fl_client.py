from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import flwr as fl

from src.model import build_model
from src.train_utils import train_one_epoch, evaluate

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def load_split(split_path: Path) -> List[List[int]]:
    return json.loads(Path(split_path).read_text(encoding="utf-8"))["clients"]

def build_dataloaders(client_indices: List[int], batch_size: int, data_dir: str = "data") -> Tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    trainset = datasets.CIFAR10(data_dir, train=True, download=False, transform=tf)
    testset  = datasets.CIFAR10(data_dir, train=False, download=False, transform=tf)

    subset = Subset(trainset, client_indices)
    pin = torch.cuda.is_available()
    # На Windows иногда быстрее num_workers=0. Поставим 0 для стабильности.
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, test_loader

def get_params(model: torch.nn.Module):
    return [t.detach().cpu().numpy() for t in model.state_dict().values()]

def set_params(model: torch.nn.Module, params: List[np.ndarray]):
    sd = model.state_dict()
    assert len(params) == len(sd)
    new_sd = {k: torch.tensor(v) for (k, v) in zip(sd.keys(), params)}
    model.load_state_dict(new_sd, strict=True)

class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid: str, indices: List[int], device, lr: float, local_epochs: int, batch_size: int):
        self.cid = cid
        self.device = device
        self.model = build_model().to(device)
        self.indices = indices
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.train_loader, self.test_loader = build_dataloaders(indices, batch_size)

    def get_parameters(self, config: Dict[str, str]):
        return get_params(self.model)

    def fit(self, parameters, config: Dict[str, str]):
        set_params(self.model, parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        metrics = {}
        for _ in range(self.local_epochs):
            metrics = train_one_epoch(self.model, self.train_loader, self.device, optimizer, criterion)
        num_examples = len(self.train_loader.dataset)
        return get_params(self.model), num_examples, metrics

    def evaluate(self, parameters, config: Dict[str, str]):
        set_params(self.model, parameters)
        m = evaluate(self.model, self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        return float(m["val_loss"]), num_examples, {"val_acc": float(m["val_acc"])}