from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Any, Tuple, Dict

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, Partitioner
from flwr_datasets.visualization import plot_label_distributions

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SplitScheme = Literal["iid", "dirichlet"]
Transform = Callable[[Any], Any]


# ============================================================
# Default hyperparameters (moved from pyproject.toml -> task.py)
# ============================================================

@dataclass(frozen=True)
class TrainHParams:
    """Дефолтные гиперпараметры обучения (используются, если их нет в run_config)."""
    batch_size: int = 64
    local_epochs: int = 5
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0
    num_workers: int = 0


def default_hparams(dataset: str) -> TrainHParams:
    """
    Вернуть дефолтные гиперпараметры для датасета.
    Сейчас оставлены значения как в твоём pyproject.toml (batch=64, epochs=5, lr=0.001).
    При желании можно сделать dataset-specific (MNIST/CIFAR) — просто раскомментируй/измени.
    """
    _ = dataset.lower()
    # пример (если захочешь разные дефолты):
    # if _.startswith("mnist"):
    #     return TrainHParams(batch_size=64, local_epochs=5, lr=0.01)
    # if _.startswith("cifar"):
    #     return TrainHParams(batch_size=64, local_epochs=5, lr=0.05)
    return TrainHParams()


# ============================================================


@dataclass(frozen=True)
class FederatedData:
    dataset_name: str
    num_clients: int
    scheme: SplitScheme
    alpha: Optional[float]
    seed: int

    train: Dataset
    test: Optional[Dataset]
    label_name: str

    partitioner: Partitioner
    get_partition: Callable[[int], Dataset]

    plot_path: Optional[Path]
    distribution_df: object  # обычно pandas.DataFrame


def _infer_label_name(ds: Dataset) -> str:
    keys = set(ds.features.keys())
    if "label" in keys:
        return "label"
    if "labels" in keys:
        return "labels"
    raise KeyError(f"Не нашёл колонку меток. Доступные колонки: {sorted(keys)}")


def _infer_image_name(ds: Dataset) -> str:
    keys = set(ds.features.keys())
    for k in ("img", "image", "pixel_values"):
        if k in keys:
            return k
    raise KeyError(f"Не нашёл колонку с изображением. Доступные: {sorted(keys)}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Единая функция выбора девайса."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_dataset_anywhere(dataset: str, data_dir: str = "data") -> DatasetDict:
    """
    Если существует локальная папка (dataset или data_dir/dataset) -> load_from_disk.
    Иначе грузим с HF Hub через load_dataset(dataset).
    """
    p = Path(dataset)
    if p.exists() and p.is_dir():
        return load_from_disk(str(p))

    p2 = Path(data_dir) / dataset
    if p2.exists() and p2.is_dir():
        return load_from_disk(str(p2))

    return load_dataset(dataset)


def prepare_federated_dataset(
    dataset: str,
    num_clients: int,
    scheme: SplitScheme = "iid",
    alpha: float = 0.3,
    seed: int = 42,
    split_train: str = "train",
    split_test: str = "test",
    data_dir: str = "data",
    min_partition_size: int = 0,
    save_plot_to: Optional[str] = None,
    plot_type: Literal["bar", "heatmap"] = "bar",
    size_unit: Literal["absolute", "percent"] = "absolute",
) -> FederatedData:
    """
    1) Загружает датасет (локально из data/.. или с HF Hub)
    2) Делит train на num_clients по IID или Dirichlet(alpha)
    3) Строит визуализацию распределения меток и сохраняет PNG (если save_plot_to задан)
    """
    ds_dict = _load_dataset_anywhere(dataset, data_dir=data_dir)

    if split_train not in ds_dict:
        raise KeyError(f"Split '{split_train}' не найден. Доступные: {list(ds_dict.keys())}")

    train = ds_dict[split_train]
    test = ds_dict[split_test] if split_test in ds_dict else None
    label_name = _infer_label_name(train)

    if scheme == "iid":
        partitioner: Partitioner = IidPartitioner(num_partitions=num_clients)
        alpha_used: Optional[float] = None
    elif scheme == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=label_name,
            alpha=alpha,
            seed=seed,
            min_partition_size=min_partition_size,
        )
        alpha_used = alpha
    else:
        raise ValueError(f"Unknown scheme='{scheme}'")

    # Для локальных данных: присваиваем датасет партишенеру.
    # Разбиение произойдёт при первом load_partition().
    partitioner.dataset = train

    def get_partition(cid: int) -> Dataset:
        if cid < 0 or cid >= num_clients:
            raise ValueError(f"cid должен быть в диапазоне [0, {num_clients-1}], получено: {cid}")
        return partitioner.load_partition(partition_id=cid)

    plot_path = None
    df = None
    if save_plot_to:
        plot_path = Path(save_plot_to)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax, df = plot_label_distributions(
            partitioner=partitioner,
            label_name=label_name,
            plot_type=plot_type,
            size_unit=size_unit,
            legend=True,
            verbose_labels=True,
            title=f"{dataset}: label distribution per client ({scheme})",
        )
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")

        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    return FederatedData(
        dataset_name=dataset,
        num_clients=num_clients,
        scheme=scheme,
        alpha=alpha_used,
        seed=seed,
        train=train,
        test=test,
        label_name=label_name,
        partitioner=partitioner,
        get_partition=get_partition,
        plot_path=plot_path,
        distribution_df=df,
    )


def make_client_trainloader(
    fed: FederatedData,
    cid: int,
    train_transform: Optional[Transform],
    batch_size: int,
    *,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """Клиент: только train (партиция клиента cid)."""
    import torch
    from torchvision.transforms import ToTensor

    part = fed.get_partition(cid)
    img_col = _infer_image_name(part)
    label_col = fed.label_name

    if train_transform is None:
        train_transform = ToTensor()

    def apply_train(batch):
        batch[img_col] = [train_transform(x) for x in batch[img_col]]
        return batch

    part = part.with_transform(apply_train)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        part,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    meta = {"img_col": img_col, "label_col": label_col}
    return train_loader, meta


def make_server_testloader(
    fed: FederatedData,
    test_transform: Optional[Transform],
    batch_size: int,
    *,
    num_workers: int = 0,
):
    """Сервер: общий test сплит (централизованная оценка)."""
    import torch
    from torchvision.transforms import ToTensor

    if fed.test is None:
        raise ValueError("В fed.test нет тестового сплита. Проверь prepare_federated_dataset(...).")

    test_ds = fed.test
    img_col = _infer_image_name(test_ds)
    label_col = fed.label_name

    if test_transform is None:
        test_transform = ToTensor()

    def apply_test(batch):
        batch[img_col] = [test_transform(x) for x in batch[img_col]]
        return batch

    test_ds = test_ds.with_transform(apply_test)

    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    meta = {"img_col": img_col, "label_col": label_col}
    return test_loader, meta


def create_model(dataset: str) -> nn.Module:
    """
    Одна и та же модель для MNIST и CIFAR-10.
    - MNIST: 1 канал
    - CIFAR-10: 3 канала
    """
    dataset = dataset.lower()
    in_channels = 1 if dataset.startswith("mnist") else 3
    num_classes = 10

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


def train_one_client(
    model: nn.Module,
    trainloader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    img_col: str,
    label_col: str,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> float:
    """Клиентское обучение. Возвращает средний train loss."""
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            x = batch[img_col].to(device)
            y = batch[label_col].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_global(
    model: nn.Module,
    testloader,
    *,
    device: torch.device,
    img_col: str,
    label_col: str,
) -> Tuple[float, float]:
    """Серверная оценка на общем test. Возвращает (loss, accuracy)."""
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    batches = 0
    correct = 0
    total = 0

    for batch in testloader:
        x = batch[img_col].to(device)
        y = batch[label_col].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        batches += 1

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    loss_avg = total_loss / max(batches, 1)
    acc = correct / max(total, 1)
    return loss_avg, acc
