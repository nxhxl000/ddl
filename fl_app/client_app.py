"""Flower ClientApp — тонкая обёртка вокруг local_train.

Поддерживает FedAvg/FedAvgM/FedProx/FedNovaM одной веткой кода:
- FedProx активируется, если в config пришло "proximal-mu" > 0.

Клиентские гиперпараметры читаются из run_config (pyproject.toml).
"""

from __future__ import annotations

import time
from pathlib import Path

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_app.data import build_loader
from fl_app.models import build_model, get_hparams
from fl_app.profiling import collect_data_profile
from fl_app.training import get_device, local_train

app = ClientApp()


def _partition_dir(rc, node_config) -> Path:
    pid = int(node_config["partition-id"])
    return Path(rc.get("data-dir", "data/")) / "partitions" / rc["partition-name"] / f"client_{pid}"


def _hp(rc, model_hp: dict, agg: str, key: str, default=None):
    """Резолвинг гиперпараметра:
      1. run_config per-strategy override: `{agg}-{key}`
      2. run_config global override: `{key}`
      3. model defaults (с учётом per-strategy): model_hp[key]
      4. hardcoded default
    """
    if f"{agg}-{key}" in rc:
        return rc[f"{agg}-{key}"]
    if key in rc:
        return rc[key]
    if key in model_hp:
        return model_hp[key]
    return default


@app.train()
def train(msg: Message, context: Context) -> Message:
    rc = context.run_config
    agg = str(rc.get("aggregation", "fedavg")).lower()
    model_name = rc["model"]
    model_hp = get_hparams(model_name, agg)
    device = get_device()
    model = build_model(model_name)

    pid = int(context.node_config["partition-id"])
    cfg_in = msg.content["config"]
    excluded = str(cfg_in.get("excluded-clients", "") or rc.get("excluded-clients", "")).strip()
    if excluded and pid in {int(x) for x in excluded.split(",")}:
        # Schema MUST идентично с обычным reply, иначе flwr InconsistentMessageReplies
        return Message(
            content=RecordDict({
                "arrays": msg.content["arrays"],
                "metrics": MetricRecord({
                    "partition-id":     float(pid),
                    "num-examples":     0.0,
                    "num-steps":        0.0,
                    "train-loss-first": 0.0,
                    "train-loss-last":  0.0,
                    "t-compute":        0.0,
                    "t-serialize":      0.0,
                    "w-drift":          0.0,
                    "update-norm-rel":  0.0,
                    "grad-norm-last":   0.0,
                    "chunk-fraction":   1.0,
                    "local-epochs":     0.0,
                }),
            }),
            reply_to=msg,
        )

    epochs = int(_hp(rc, model_hp, agg, "local-epochs"))
    lr = float(_hp(rc, model_hp, agg, "client-lr"))
    momentum = float(_hp(rc, model_hp, agg, "client-momentum"))
    wd = float(_hp(rc, model_hp, agg, "client-weight-decay"))
    bs = int(_hp(rc, model_hp, agg, "batch-size"))
    opt_name = str(_hp(rc, model_hp, agg, "optimizer")).lower()

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    cfg = msg.content["config"]
    proximal_mu = float(cfg.get("proximal-mu", 0.0))
    lr = lr * float(cfg.get("lr-scale", 1.0))

    # per-client-chunks: cfg (per-round, динамический schedule) → rc (статика) → дефолт
    per_client = str(cfg.get("per-client-chunks", "") or rc.get("per-client-chunks", "")).strip()
    if per_client:
        parts = [float(x) for x in per_client.split(",")]
        chunk_fraction = parts[pid] if pid < len(parts) else 1.0
    else:
        chunk_fraction = float(_hp(rc, model_hp, agg, "chunk-fraction", 1.0))

    # per-client-epochs: аналогично, override локального epochs если задано в cfg/rc
    per_client_ep = str(cfg.get("per-client-epochs", "") or rc.get("per-client-epochs", "")).strip()
    if per_client_ep:
        parts_ep = [int(x) for x in per_client_ep.split(",")]
        if pid < len(parts_ep):
            epochs = parts_ep[pid]
    server_round = int(cfg.get("server-round", 0))
    loader = build_loader(
        _partition_dir(rc, context.node_config),
        batch_size=bs,
        train=True,
        chunk_fraction=chunk_fraction,
        chunk_seed=server_round * 100 + pid,
    )

    res = local_train(
        model, loader,
        lr=lr, momentum=momentum, weight_decay=wd,
        epochs=epochs, device=device,
        proximal_mu=proximal_mu,
        optimizer=opt_name,
    )

    t_serialize_start = time.time()
    reply_arrays = ArrayRecord(model.state_dict())
    t_serialize = time.time() - t_serialize_start

    metrics_dict: dict[str, float] = {
        "partition-id":     float(pid),
        "num-examples":     float(res["num_examples"]),
        "num-steps":        float(res["num_steps"]),
        "train-loss-first": float(res["loss_first"]),
        "train-loss-last":  float(res["loss_last"]),
        "t-compute":        float(res["t_compute"]),
        "t-serialize":      float(t_serialize),
        "w-drift":          float(res["w_drift"]),
        "update-norm-rel":  float(res["update_norm_rel"]),
        "grad-norm-last":   float(res["grad_norm_last"]),
        "chunk-fraction":   float(chunk_fraction),
        "local-epochs":     float(epochs),
    }
    # Round 1: класс-распределение для серверного подсчёта MPJS/Gini.
    # data_cls_{N} = число сэмплов класса N. Сервер фильтрует ключи перед aggregate.
    if server_round == 1:
        metrics_dict.update(collect_data_profile(_partition_dir(rc, context.node_config)))
    metrics = MetricRecord(metrics_dict)
    return Message(
        content=RecordDict({"arrays": reply_arrays, "metrics": metrics}),
        reply_to=msg,
    )


@app.evaluate()
def eval_fn(msg: Message, context: Context) -> Message:
    return Message(
        content=RecordDict({"metrics": MetricRecord({"num-examples": 0.0})}),
        reply_to=msg,
    )
