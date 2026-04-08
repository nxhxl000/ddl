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
    append_index_row,
    append_rounds_row,
    generate_plots,
    init_csvs,
    log_round,
    make_exp_dir,
    print_summary_table,
    write_config,
    write_log_header,
    write_summary,
)
from fl_app.models import build_model, get_hparams
from fl_app.adaptive import compute_adaptive_params, make_adaptive_log, print_adaptive_summary, to_train_config_dict
from fl_app.profiling import print_profiling_summary, run_profiling_round, save_cluster_profile
from fl_app.server_data import (
    compute_effective_js, compute_server_schedule,
    make_server_log, print_server_schedule_summary, to_server_config_dict,
)
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
    enable_profiling:   bool  = str(rc.get("enable-profiling", "true")).lower() != "false"
    adaptive_mode:      str   = str(rc.get("adaptive-mode", "maximize-epochs"))
    server_mode:        str   = str(rc.get("server-mode", "disabled"))

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
        # adaptive
        "enable_profiling": enable_profiling,
        "adaptive_mode":    adaptive_mode if enable_profiling else "disabled",
        # server dataset
        "server_mode": server_mode,
    }

    # ── Директория и файлы эксперимента ──────────────────────────────────────
    exp_dir, exp_name = make_exp_dir(partition_name, model_name, agg_name)

    config_path  = exp_dir / "config.json"
    log_path     = exp_dir / "train.log"
    model_path   = exp_dir / "model.pt"
    summary_path = exp_dir / "summary.json"
    rounds_csv   = exp_dir / "metrics" / "rounds.csv"
    clients_csv  = exp_dir / "metrics" / "clients.csv"
    classes_csv  = exp_dir / "metrics" / "classes.csv"
    plots_dir    = exp_dir / "plots"

    # Записываем конфиг ДО начала обучения
    write_config(config_path, config=config, model=global_model, device=device)
    write_log_header(log_path, config=config, model=global_model, device=device)
    init_csvs(rounds_csv, clients_csv, classes_csv)

    # ── Профилировочный раунд (до основного обучения) ─────────────────────────
    profile_path    = exp_dir / "cluster_profile.json"
    profiles:       dict = {}
    adaptive_flat:  dict = {}
    server_schedule: dict = {}
    effective_js:   float = 0.0

    if enable_profiling:
        print("\nЗапуск профилировочного раунда...")
        profiles = run_profiling_round(
            grid=grid,
            initial_arrays=initial_arrays,
            fraction_train=fraction_train,
            min_train_nodes=min_train_nodes,
            min_available_nodes=min_available_nodes,
        )
        save_cluster_profile(
            profiles, exp_dir,
            partition_name=partition_name,
            num_classes=manifest["num_classes"],
        )
        print_profiling_summary(profiles, num_classes=manifest["num_classes"])
        print(f"  Профиль сохранён: {profile_path.name}")

        # ── Адаптивное расписание (straggler mitigation) ──────────────────────
        adaptive_params = compute_adaptive_params(profiles, base_epochs=local_epochs, mode=adaptive_mode)
        adaptive_flat   = to_train_config_dict(adaptive_params)
        print_adaptive_summary(adaptive_params, profiles, base_epochs=local_epochs, mode=adaptive_mode, tolerance=0.10)

        # Дописываем адаптивное расписание в cluster_profile.json
        adaptive_log = make_adaptive_log(adaptive_params, profiles, local_epochs, adaptive_mode, tolerance=0.10)
        profile_data = json.loads(profile_path.read_text())
        profile_data["adaptive_schedule"] = adaptive_log

        # ── Серверный датасет (если режим не disabled и server/ существует) ────
        server_dir  = part_dir / "server"
        server_size = manifest.get("server_size")
        if server_mode != "disabled" and server_size and server_dir.exists():
            print("\nВычисление расписания серверного датасета...")
            server_schedule = compute_server_schedule(
                profiles,
                num_classes=manifest["num_classes"],
                server_size=server_size,
                mode=server_mode,
            )
            js_before = profile_data.get("mean_pairwise_js", 0.0)
            effective_js = compute_effective_js(
                server_schedule, profiles, manifest["num_classes"]
            )
            print_server_schedule_summary(
                server_schedule, profiles,
                num_classes=manifest["num_classes"],
                server_size=server_size,
                mode=server_mode,
                class_names=manifest["class_names"],
                mean_pairwise_js_before=js_before,
            )
            profile_data["server_schedule"] = make_server_log(
                server_schedule, profiles,
                num_classes=manifest["num_classes"],
                server_size=server_size,
                mode=server_mode,
                mean_pairwise_js_before=js_before,
            )
        elif server_mode != "disabled":
            if not server_size:
                print("\n[server-data] server-mode задан, но партиция не содержит server/ (server_size=None в manifest). Пропускаем.")
            elif not server_dir.exists():
                print(f"\n[server-data] server-mode задан, но {server_dir} не найден. Пропускаем.")

        profile_path.write_text(json.dumps(profile_data, indent=2, ensure_ascii=False))
        print()

    # ── Состояние FL-цикла ────────────────────────────────────────────────────
    run_start = time.time()
    cum_comm_mb = 0.0
    prev_acc = 0.0
    all_round_accs: List[Tuple[int, float]] = []
    all_round_f1s:  List[Tuple[int, float]] = []

    # ── Callback оценки после каждого раунда ─────────────────────────────────
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        nonlocal cum_comm_mb, prev_acc

        wall_clock = time.time() - run_start

        eval_start = time.time()
        model = build_model(model_name)
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, acc, per_class, f1 = evaluate(
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
                  test_acc=acc, test_f1=f1, test_loss=loss,
                  train_time=train_time, agg_time=agg_time, eval_time=eval_time,
                  effective_js=effective_js)
        cum_comm_mb = append_rounds_row(
            rounds_csv, server_round=server_round, wall_clock_sec=wall_clock,
            acc=acc, f1=f1, delta_acc=delta_acc,
            loss=loss, client_logs=client_logs, model_bytes=model_bytes,
            train_time_sec=train_time, agg_time_sec=agg_time, eval_time_sec=eval_time,
            cum_comm_mb=cum_comm_mb, effective_js=effective_js,
        )
        append_client_rows(clients_csv, server_round=server_round, client_logs=client_logs)
        append_classes_rows(
            classes_csv, server_round=server_round,
            per_class=per_class, class_names=manifest["class_names"],
        )
        all_round_accs.append((server_round, acc))
        all_round_f1s.append((server_round, f1))

        if server_round == num_rounds:
            write_summary(
                summary_path, exp_name=exp_name, all_round_accs=all_round_accs,
                all_round_f1s=all_round_f1s, total_wall_time=time.time() - run_start,
                cum_comm_mb=cum_comm_mb, config=config,
            )

        return MetricRecord({"test_loss": float(loss), "test_acc": float(acc), "test_f1": float(f1)})

    # ── Собираем статичный train_config ───────────────────────────────────────
    server_flat: dict = {}
    if server_schedule:
        for pid, counts in server_schedule.items():
            for k, cnt in enumerate(counts):
                server_flat[f"c{pid}_srv_{k}"] = float(cnt)

    # ── Запуск FL ─────────────────────────────────────────────────────────────
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"learning-rate": hp.lr, **adaptive_flat, **server_flat}),
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
        plots_dir=plots_dir,
        class_names=manifest["class_names"],
        exp_name=exp_name,
    )

    # ── Запись в index.csv ────────────────────────────────────────────────────
    best_round, best_acc = max(all_round_accs, key=lambda x: x[1])
    best_f1 = max(all_round_f1s, key=lambda x: x[1])[1] if all_round_f1s else 0.0
    append_index_row(
        "experiments",
        exp_name=exp_name,
        config=config,
        best_acc=best_acc,
        best_f1=best_f1,
        best_round=best_round,
        num_rounds=num_rounds,
        total_time=total_wall_time,
    )

    # ── Итоговый вывод ────────────────────────────────────────────────────────
    print(f"Эксперимент : {exp_name}")
    print(f"Папка       : {exp_dir.resolve()}")
    print(f"Артефакты   :")
    print(f"  config.json")
    print(f"  model.pt")
    print(f"  train.log")
    print(f"  summary.json")
    print(f"  metrics/rounds.csv")
    print(f"  metrics/clients.csv")
    print(f"  metrics/classes.csv")
    if enable_profiling:
        print(f"  cluster_profile.json")
    for p in sorted(plots, key=lambda x: x.name):
        print(f"  plots/{p.name}")
