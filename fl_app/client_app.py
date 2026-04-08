from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from datasets import load_from_disk, load_dataset

from fl_app.models import build_model, get_hparams
from fl_app.profiling import collect_data_profile, collect_hardware_info, run_benchmark
from fl_app.server_data import sample_server_dataset
from fl_app.training import get_device, local_train, make_dataloader, stratified_select

app = ClientApp()


def _load_production_dataset(data_dir: str):
    """Загрузить данные клиента из imagefolder (class_name/img.jpg)."""
    data_path = Path(data_dir)
    # Проверяем: если это HF Dataset на диске (есть dataset_info.json), грузим напрямую
    if (data_path / "dataset_info.json").exists():
        return load_from_disk(str(data_path))
    # Иначе — imagefolder (папки с классами)
    ds = load_dataset("imagefolder", data_dir=str(data_path), split="train")
    return ds


@app.train()
def train(msg: Message, context: Context) -> Message:
    rc = context.run_config
    data_mode:      str = rc.get("data-mode", "research")
    model_name:     str = rc["model"]
    local_epochs:   int = int(rc.get("local-epochs", 5))
    data_dir:       str = rc.get("data-dir", "data/")

    # Загружаем веса от сервера
    model = build_model(model_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    # ── Определяем путь к данным в зависимости от режима ─────────────────────
    if data_mode == "production":
        # Production: данные клиента, путь из node_config
        client_data_dir = context.node_config.get("data-dir", "data/local")
        partition_path = Path(client_data_dir)
        partition_id = 0  # в production нет partition-id
    else:
        # Research: предразбитые партиции
        partition_name: str = rc["partition-name"]
        partition_id = int(context.node_config.get("partition-id", 0))
        partition_path = Path(data_dir) / "partitions" / partition_name / f"client_{partition_id}"

    # ── Профилировочный раунд ─────────────────────────────────────────────────
    try:
        is_profiling = float(msg.content["config"]["profiling-mode"]) == 1.0
    except Exception:
        is_profiling = False

    if is_profiling:
        benchmark_samples = int(msg.content["config"].get("benchmark-samples", 1000.0))
        benchmark_epochs  = int(msg.content["config"].get("benchmark-epochs",  2.0))

        hw      = collect_hardware_info()
        data    = collect_data_profile(partition_path)
        bench   = run_benchmark(
            copy.deepcopy(model), partition_path, device,
            max_samples=benchmark_samples,
            epochs=benchmark_epochs,
        )

        metrics = MetricRecord({
            "partition-id": float(partition_id),
            "num-examples": float(benchmark_samples),
            **hw,
            **data,
            **bench,
        })
        arrays = ArrayRecord(model.state_dict())
        return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)

    # ── Обычный тренировочный раунд ───────────────────────────────────────────
    hp = get_hparams(model_name)

    try:
        mu = float(msg.content["config"]["proximal-mu"])
    except Exception:
        mu = 0.0

    # ── SCAFFOLD: загрузка c_server и c_i ─────────────────────────────────────
    c_server_state = None
    c_i_state      = None
    if "c_server" in msg.content:
        c_server_state = msg.content["c_server"].to_torch_state_dict()
        if "c_client" in context.state:
            c_i_state = context.state["c_client"].to_torch_state_dict()
        else:
            c_i_state = {k: torch.zeros_like(v) for k, v in c_server_state.items()}

    # ── Адаптивные параметры (из профилировочного раунда) ─────────────────────
    cfg = msg.content["config"]
    try:
        adaptive_epochs = int(cfg[f"c{partition_id}_epochs"])
        if adaptive_epochs > 0:
            local_epochs = adaptive_epochs
    except Exception:
        pass

    sample_budget = -1
    try:
        sample_budget = int(cfg[f"c{partition_id}_samples"])
    except Exception:
        pass

    try:
        server_round = int(cfg["server-round"])
    except Exception:
        server_round = 0

    # ── Загрузка датасета ─────────────────────────────────────────────────────
    if data_mode == "production":
        # Production: загружаем данные клиента
        ds = _load_production_dataset(str(partition_path))
    elif sample_budget > 0:
        # Research: стратифицированный чанкинг
        shuffle_seed = server_round * 100 + partition_id
        ds = load_from_disk(str(partition_path))
        lc = next(c for c in ("label", "labels", "fine_label", "coarse_label") if c in ds.features)
        ds = stratified_select(ds, sample_budget, lc, seed=shuffle_seed)
    else:
        ds = None  # make_dataloader загрузит с диска сам

    # ── Серверный датасет (только research mode) ──────────────────────────────
    server_ds = None
    if data_mode == "research":
        num_classes = max(
            (int(k.split("_srv_")[1]) for k in cfg if k.startswith(f"c{partition_id}_srv_")),
            default=-1,
        ) + 1
        if num_classes > 0:
            server_path = Path(data_dir) / "partitions" / rc["partition-name"] / "server"
            if server_path.exists():
                class_counts = [
                    int(cfg.get(f"c{partition_id}_srv_{k}", 0.0))
                    for k in range(num_classes)
                ]
                round_seed = server_round * 997
                server_ds = sample_server_dataset(server_path, class_counts, round_seed)

    # ── Объединяем локальные + серверные данные ────────────────────────────────
    if server_ds is not None and len(server_ds) > 0:
        from datasets import concatenate_datasets
        local_ds = ds if ds is not None else load_from_disk(str(partition_path))
        combined_ds = concatenate_datasets([local_ds, server_ds])
        loader, img_col, label_col = make_dataloader(
            partition_path, hp.batch_size,
            shuffle=True, num_workers=hp.num_workers, augment=True, dataset=combined_ds,
        )
    elif ds is not None:
        loader, img_col, label_col = make_dataloader(
            partition_path, hp.batch_size,
            shuffle=True, num_workers=hp.num_workers, augment=True, dataset=ds,
        )
    else:
        loader, img_col, label_col = make_dataloader(
            partition_path, hp.batch_size, shuffle=True, num_workers=hp.num_workers, augment=True
        )

    # Снапшот весов до обучения (нужен для SCAFFOLD c_i_new)
    x0_state = {k: v.clone() for k, v in model.state_dict().items()} if c_server_state else None

    global_weights = [p.data.clone() for p in model.parameters()]

    t0 = time.perf_counter()
    epoch_losses, num_examples, total_steps = local_train(
        model, loader,
        device=device,
        epochs=local_epochs,
        lr=hp.lr,
        momentum=hp.momentum,
        weight_decay=hp.weight_decay,
        img_col=img_col,
        label_col=label_col,
        mu=mu,
        c_i=c_i_state,
        c_server=c_server_state,
    )
    round_time = time.perf_counter() - t0

    drift = torch.sqrt(sum(
        (p.data.cpu() - wg).pow(2).sum()
        for p, wg in zip(model.parameters(), global_weights)
    )).item()

    # ── SCAFFOLD: вычисление c_i_new и delta_c ────────────────────────────────
    c_delta_arrays = None
    if c_server_state is not None and x0_state is not None and total_steps > 0:
        y_i_state = model.state_dict()
        K, lr = total_steps, hp.lr
        c_i_new_state = {
            name: (
                c_i_state[name]
                - c_server_state[name]
                + (x0_state[name].to(device) - y_i_state[name]) / (K * lr)
            )
            for name in c_i_state
        }
        context.state["c_client"] = ArrayRecord(c_i_new_state)
        c_delta_arrays = ArrayRecord({
            name: c_i_new_state[name] - c_i_state[name]
            for name in c_i_state
        })

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
    content = RecordDict({"arrays": arrays, "metrics": metrics})
    if c_delta_arrays is not None:
        content["c_delta"] = c_delta_arrays
    return Message(content=content, reply_to=msg)
