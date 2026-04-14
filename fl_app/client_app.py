"""Flower ClientApp — тонкая обёртка вокруг local_train.

Поддерживает FedAvg/FedAvgM/FedProx/SCAFFOLD одной веткой кода:
- FedProx активируется, если в config пришло "proximal-mu" > 0.
- SCAFFOLD активируется, если в payload есть ArrayRecord "c_server".
  c_i хранится в context.state["c_client"] между раундами.

Клиентские гиперпараметры читаются из run_config (pyproject.toml).
"""

from __future__ import annotations

from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Array

from fl_app.data import build_loader
from fl_app.models import build_model
from fl_app.profiling import collect_data_profile, collect_hardware_info, run_benchmark
from fl_app.strategies import C_DELTA_PREFIX, C_SERVER_KEY
from fl_app.training import get_device, local_train

app = ClientApp()


def _partition_dir(rc, node_config) -> Path:
    pid = int(node_config["partition-id"])
    return Path(rc.get("data-dir", "data/")) / "partitions" / rc["partition-name"] / f"client_{pid}"


def _ar_to_dict(ar: ArrayRecord) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(a.numpy()).clone() for k, a in ar.items()}


def _hp(rc, agg: str, key: str, default):
    """Per-strategy оверрайд → глобальный → дефолт."""
    return rc.get(f"{agg}-{key}", rc.get(key, default))


@app.train()
def train(msg: Message, context: Context) -> Message:
    rc = context.run_config
    agg = str(rc.get("aggregation", "fedavg")).lower()
    device = get_device()
    model = build_model(rc["model"])

    cfg_in = msg.content["config"]
    if float(cfg_in.get("profiling-mode", 0.0)) == 1.0:
        pid = int(context.node_config["partition-id"])
        part_dir = _partition_dir(rc, context.node_config)
        hw = collect_hardware_info()
        data = collect_data_profile(part_dir)
        bench = run_benchmark(
            model, part_dir, device,
            max_samples=int(cfg_in.get("benchmark-samples", 1000)),
            epochs=int(cfg_in.get("benchmark-epochs", 2)),
            batch_size=int(_hp(rc, agg, "batch-size", 64)),
        )
        metrics = MetricRecord({"partition-id": float(pid), **hw, **data, **bench})
        return Message(
            content=RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics}),
            reply_to=msg,
        )

    epochs = int(_hp(rc, agg, "local-epochs", 2))
    lr = float(_hp(rc, agg, "client-lr", 0.03))
    momentum = float(_hp(rc, agg, "client-momentum", 0.9))
    wd = float(_hp(rc, agg, "client-weight-decay", 5e-4))
    bs = int(_hp(rc, agg, "batch-size", 64))

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    loader = build_loader(
        _partition_dir(rc, context.node_config),
        batch_size=bs,
        train=True,
    )

    cfg = msg.content["config"]
    proximal_mu = float(cfg.get("proximal-mu", 0.0))
    lr = lr * float(cfg.get("lr-scale", 1.0))

    c_server = c_client = None
    if C_SERVER_KEY in msg.content:
        c_server = _ar_to_dict(msg.content[C_SERVER_KEY])
        c_client = (
            _ar_to_dict(context.state["c_client"])
            if "c_client" in context.state
            else {n: torch.zeros_like(v) for n, v in c_server.items()}
        )

    res = local_train(
        model, loader,
        lr=lr, momentum=momentum, weight_decay=wd,
        epochs=epochs, device=device,
        proximal_mu=proximal_mu,
        c_server=c_server, c_client=c_client,
    )

    reply_arrays = ArrayRecord(model.state_dict())
    if "c_delta" in res:
        for k, v in res["c_delta"].items():
            reply_arrays[C_DELTA_PREFIX + k] = Array(v.numpy())
        context.state["c_client"] = ArrayRecord(
            {k: Array(v.numpy()) for k, v in res["c_new"].items()}
        )

    metrics = MetricRecord({
        "num-examples":     float(res["num_examples"]),
        "train-loss-first": float(res["loss_first"]),
        "train-loss-last":  float(res["loss_last"]),
        "t-compute":        float(res["t_compute"]),
        "w-drift":          float(res["w_drift"]),
        "update-norm-rel":  float(res["update_norm_rel"]),
        "grad-norm-last":   float(res["grad_norm_last"]),
    })
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
