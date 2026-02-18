from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, List

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Поддержим оба варианта: task.py в корне или fl_app/task.py
try:
    from fl_app.task import (
        create_model,
        get_device,
        prepare_federated_dataset,
        make_client_trainloader,
    )
except Exception:  # pragma: no cover
    from task import (  # type: ignore
        create_model,
        get_device,
        prepare_federated_dataset,
        make_client_trainloader,
    )


def _rc(context: Context, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            return context.run_config[k]
        except KeyError:
            pass
    return default


def _nc(context: Context, key: str, default: Any = None) -> Any:
    try:
        return context.node_config[key]
    except KeyError:
        return default


@lru_cache(maxsize=8)
def _cached_fed(
    dataset: str,
    num_clients: int,
    scheme: str,
    alpha: float,
    seed: int,
    data_dir: str,
    min_partition_size: int,
):
    return prepare_federated_dataset(
        dataset=dataset,
        num_clients=num_clients,
        scheme=scheme,
        alpha=alpha,
        seed=seed,
        data_dir=data_dir,
        split_train="train",
        split_test="test",
        min_partition_size=min_partition_size,
    )


app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    # ---- run config ----
    dataset_name: str = _rc(context, "dataset", "dataset_name", default="cifar10")
    scheme: str = _rc(context, "scheme", "split_scheme", default="iid")  # iid/dirichlet
    alpha: float = float(_rc(context, "alpha", default=0.3))
    seed: int = int(_rc(context, "seed", default=42))
    data_dir: str = _rc(context, "data-dir", "data_dir", default="data/")

    batch_size: int = int(_rc(context, "batch-size", "batch_size", default=64))
    local_epochs: int = int(_rc(context, "local-epochs", "local_epochs", default=1))
    min_partition_size: int = int(_rc(context, "min-partition-size", "min_partition_size", default=0))
    num_workers: int = int(_rc(context, "num-workers", "num_workers", default=0))

    momentum: float = float(_rc(context, "momentum", default=0.9))
    weight_decay: float = float(_rc(context, "weight_decay", default=0.0))

    # lr может приходить от сервера (train_config)
    lr_default: float = float(_rc(context, "learning-rate", "lr", default=0.001))
    try:
        lr: float = float(msg.content["config"]["learning-rate"])
    except Exception:
        try:
            lr = float(msg.content["config"]["lr"])
        except Exception:
            lr = lr_default

    # ---- node config ----
    partition_id = int(_nc(context, "partition-id", 0))
    num_partitions = int(_nc(context, "num-partitions", 1))

    # ---- model ----
    model = create_model(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = get_device(prefer_cuda=True)
    model.to(device)

    # ---- data ----
    fed = _cached_fed(
        dataset=dataset_name,
        num_clients=num_partitions,
        scheme=scheme,
        alpha=alpha,
        seed=seed,
        data_dir=data_dir,
        min_partition_size=min_partition_size,
    )

    trainloader, meta = make_client_trainloader(
        fed=fed,
        cid=partition_id,
        train_transform=None,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    img_col = meta["img_col"]
    label_col = meta["label_col"]

    # ---- epoch-wise training ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    epoch_losses: List[float] = []

    t0 = time.perf_counter()
    model.train()

    for ep in range(local_epochs):
        loss_sum = 0.0
        batches = 0

        for batch in trainloader:
            x = batch[img_col].to(device)
            y = batch[label_col].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            batches += 1

        epoch_loss = loss_sum / max(batches, 1)
        epoch_losses.append(epoch_loss)

    round_time_sec = time.perf_counter() - t0

    avg_train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
    num_examples = int(len(trainloader.dataset))

    # ---- reply ----
    arrays = ArrayRecord(model.state_dict())

    # Для стандартной агрегации FedAvg оставляем простые метрики
    metrics = MetricRecord(
        {
            "train_loss": avg_train_loss,
            "num-examples": num_examples,  # ключ для взвешивания (как у тебя в логах)
        }
    )

    # Детализация для логов (НЕ агрегируемая стратегией по умолчанию)
    details = ConfigRecord(
        {
            "partition-id": partition_id,
            "local-epochs": local_epochs,
            "epoch-train-losses": [float(x) for x in epoch_losses],
            "round-time-sec": float(round_time_sec),
            "num-examples": num_examples,
        }
    )

    content = RecordDict({"arrays": arrays, "metrics": metrics, "details": details})
    return Message(content=content, reply_to=msg)
