from __future__ import annotations

import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_app.models import build_model, get_hparams
from fl_app.training import get_device, local_train, make_dataloader

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    rc = context.run_config
    partition_name: str = rc["partition-name"]
    model_name:     str = rc["model"]
    local_epochs:   int = int(rc.get("local-epochs", 5))
    data_dir:       str = rc.get("data-dir", "data/")

    partition_id = int(context.node_config.get("partition-id", 0))
    partition_path = Path(data_dir) / "partitions" / partition_name / f"client_{partition_id}"

    # Гиперпараметры из реестра моделей
    hp = get_hparams(model_name)

    # Загружаем веса от сервера
    model = build_model(model_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # FedProx mu (приходит от сервера если стратегия = fedprox, иначе 0)
    try:
        mu = float(msg.content["config"]["proximal-mu"])
    except Exception:
        mu = 0.0

    device = get_device()
    loader, img_col, label_col = make_dataloader(
        partition_path, hp.batch_size, shuffle=True, num_workers=hp.num_workers
    )

    # Сохраняем глобальные веса до обучения — для вычисления drift
    global_weights = [p.data.clone() for p in model.parameters()]

    t0 = time.perf_counter()
    epoch_losses, num_examples = local_train(
        model, loader,
        device=device,
        epochs=local_epochs,
        lr=hp.lr,
        momentum=hp.momentum,
        weight_decay=hp.weight_decay,
        img_col=img_col,
        label_col=label_col,
        mu=mu,
    )
    round_time = time.perf_counter() - t0

    # Client drift: ||w_local - w_global||_F
    drift = torch.sqrt(sum(
        (p.data.cpu() - wg).pow(2).sum()
        for p, wg in zip(model.parameters(), global_weights)
    )).item()

    arrays  = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss":       float(sum(epoch_losses) / max(len(epoch_losses), 1)),
        "num-examples":     float(num_examples),
        "partition-id":     float(partition_id),
        "local-epochs":     float(local_epochs),
        "round-time-sec":   float(round_time),
        "first-epoch-loss": float(epoch_losses[0])  if epoch_losses else 0.0,
        "last-epoch-loss":  float(epoch_losses[-1]) if epoch_losses else 0.0,
        "drift":            float(drift),
    })
    return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)
