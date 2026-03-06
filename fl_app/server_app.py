from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fl_app.artifacts import (
    append_classes_rows,
    append_client_rows,
    append_rounds_row,
    generate_plots,
    init_csvs,
    log_round,
    make_exp_dir,
    print_summary_table,
    write_log_header,
    write_summary,
)
from fl_app.models import build_model, get_hparams
from fl_app.strategies import build_strategy
from fl_app.training import evaluate, get_device, make_dataloader

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config

    # ── Config из pyproject ───────────────────────────────────────────────────
    partition_name:     str   = rc["partition-name"]
    model_name:         str   = rc["model"]
    agg_name:           str   = rc["aggregation"]
    num_rounds:         int   = int(rc.get("num-server-rounds", 10))
    fraction_train:     float = float(rc.get("fraction-train", 1.0))
    min_train_nodes:    int   = int(rc.get("min-train-nodes", 1))
    min_available_nodes: int  = int(rc.get("min-available-nodes", 1))
    local_epochs:       int   = int(rc.get("local-epochs", 5))
    data_dir:           str   = rc.get("data-dir", "data/")

    # ── Manifest — метаданные партиции ────────────────────────────────────────
    part_dir = Path(data_dir) / "partitions" / partition_name
    manifest = json.loads((part_dir / "manifest.json").read_text())

    # ── Модель + гиперпараметры ───────────────────────────────────────────────
    hp = get_hparams(model_name)
    global_model = build_model(model_name)
    model_bytes = sum(
        int(t.numel() * t.element_size())
        for t in global_model.state_dict().values()
        if hasattr(t, "numel")
    )
    initial_arrays = ArrayRecord(global_model.state_dict())

    # ── Стратегия ─────────────────────────────────────────────────────────────
    strategy, strategy_params = build_strategy(
        agg_name,
        fraction_train=fraction_train,
        min_train_nodes=min_train_nodes,
        min_available_nodes=min_available_nodes,
    )

    # ── Тестовый загрузчик ────────────────────────────────────────────────────
    device = get_device()
    testloader, img_col, label_col = make_dataloader(
        part_dir / "test", hp.batch_size, shuffle=False, num_workers=hp.num_workers
    )

    # ── Собранный конфиг эксперимента (для артефактов) ────────────────────────
    config = {
        # pyproject
        "partition_name":      partition_name,
        "model":               model_name,
        "aggregation":         agg_name,
        "num_rounds":          num_rounds,
        "fraction_train":      fraction_train,
        "min_train_nodes":     min_train_nodes,
        "min_available_nodes": min_available_nodes,
        # manifest
        "dataset":     manifest["dataset"],
        "scheme":      manifest["scheme"],
        "alpha":       manifest.get("alpha"),
        "num_clients": manifest["num_clients"],
        "num_classes": manifest["num_classes"],
        "class_names": manifest["class_names"],
        "test_size":   manifest.get("test_size"),
        # hparams (из models.py + pyproject.toml)
        "lr":           hp.lr,
        "batch_size":   hp.batch_size,
        "local_epochs": local_epochs,
        "momentum":     hp.momentum,
        "weight_decay": hp.weight_decay,
        "num_workers":  hp.num_workers,
        # strategy params (из strategies.py)
        **{f"strategy_{k}": v for k, v in strategy_params.items()},
    }

    # ── Директория и файлы эксперимента ──────────────────────────────────────
    exp_dir, exp_name = make_exp_dir(partition_name, model_name, agg_name)
    prefix       = exp_dir / f"{num_rounds}r"
    log_path     = Path(f"{prefix}.log")
    model_path   = Path(f"{prefix}.pt")
    rounds_csv   = Path(f"{prefix}__rounds.csv")
    clients_csv  = Path(f"{prefix}__clients.csv")
    classes_csv  = Path(f"{prefix}__classes.csv")
    summary_path = Path(f"{prefix}__summary.json")

    write_log_header(log_path, config=config, model=global_model, device=device)
    init_csvs(rounds_csv, clients_csv, classes_csv)

    # ── Состояние FL-цикла ────────────────────────────────────────────────────
    run_start = time.time()
    cum_comm_mb = 0.0
    prev_acc = 0.0
    all_round_accs: List[Tuple[int, float]] = []

    # ── Callback оценки после каждого раунда ─────────────────────────────────
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        nonlocal cum_comm_mb, prev_acc

        eval_start = time.time()
        model = build_model(model_name)
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc, per_class = evaluate(
            model, testloader, device=device, img_col=img_col, label_col=label_col
        )
        eval_time = time.time() - eval_start

        delta_acc   = acc - prev_acc
        prev_acc    = acc
        client_logs = strategy.get_round_logs(server_round)

        agg_start = strategy._agg_start_times.get(server_round, eval_start)
        agg_end   = strategy._agg_end_times.get(server_round, agg_start)
        agg_time  = agg_end - agg_start
        train_time = max(
            (v["round_time_sec"] for v in client_logs.values()), default=0.0
        )

        log_round(log_path, server_round=server_round, client_logs=client_logs,
                  test_acc=acc, test_loss=loss,
                  train_time=train_time, agg_time=agg_time, eval_time=eval_time)
        cum_comm_mb = append_rounds_row(
            rounds_csv, server_round=server_round, acc=acc, delta_acc=delta_acc,
            loss=loss, client_logs=client_logs, model_bytes=model_bytes,
            train_time_sec=train_time, agg_time_sec=agg_time, eval_time_sec=eval_time,
            cum_comm_mb=cum_comm_mb,
        )
        append_client_rows(clients_csv, server_round=server_round, client_logs=client_logs)
        append_classes_rows(
            classes_csv, server_round=server_round,
            per_class=per_class, class_names=manifest["class_names"],
        )
        all_round_accs.append((server_round, acc))

        if server_round == num_rounds:
            write_summary(
                summary_path, exp_name=exp_name, all_round_accs=all_round_accs,
                total_wall_time=time.time() - run_start,
                cum_comm_mb=cum_comm_mb, config=config,
            )

        return MetricRecord({"test_loss": float(loss), "test_acc": float(acc)})

    # ── Запуск FL ─────────────────────────────────────────────────────────────
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"learning-rate": hp.lr}),
        evaluate_fn=global_evaluate,
    )

    torch.save(result.arrays.to_torch_state_dict(), str(model_path))

    # ── Таблица результатов в терминале ───────────────────────────────────────
    total_wall_time = time.time() - run_start
    print_summary_table(
        rounds_csv, exp_name=exp_name,
        total_wall_time=total_wall_time, cum_comm_mb=cum_comm_mb,
    )

    # ── Графики ───────────────────────────────────────────────────────────────
    plots = generate_plots(
        rounds_csv, clients_csv, classes_csv,
        out_prefix=prefix,
        class_names=manifest["class_names"],
        exp_name=exp_name,
    )

    print(f"Эксперимент : {exp_name}")
    print(f"Папка       : {exp_dir.resolve()}")
    print(f"Артефакты   :")
    print(f"  {model_path.name}")
    print(f"  {log_path.name}")
    print(f"  {rounds_csv.name}")
    print(f"  {clients_csv.name}")
    print(f"  {classes_csv.name}")
    print(f"  {summary_path.name}")
    for p in plots:
        print(f"  {p.name}")
