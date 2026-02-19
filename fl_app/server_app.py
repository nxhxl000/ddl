from __future__ import annotations

import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import flwr
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp

# --- serverapp strategies (Message API) ---
from flwr.serverapp.strategy import (
    FedAvg,
    FedAvgM,
    FedAdam,
    FedYogi,
    FedAdagrad,
    FedMedian,
    FedTrimmedAvg,
    Krum,
    MultiKrum,
    Bulyan,
)

# Поддержим оба варианта: task.py в корне или fl_app/task.py
try:
    from fl_app.task import (
        create_model,
        get_device,
        prepare_federated_dataset,
        make_server_testloader,
        evaluate_global,
        default_hparams,
    )
except Exception:  # pragma: no cover
    from task import (  # type: ignore
        create_model,
        get_device,
        prepare_federated_dataset,
        make_server_testloader,
        evaluate_global,
        default_hparams,
    )

try:  # optional, но у тебя torchvision в зависимостях
    import torchvision  # type: ignore
except Exception:  # pragma: no cover
    torchvision = None  # type: ignore


def _rc(context: Context, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            return context.run_config[k]
        except KeyError:
            pass
    return default


def _run_config_to_dict(context: Context) -> Dict[str, Any]:
    rc: Dict[str, Any] = {}
    try:
        for k in context.run_config:
            rc[str(k)] = context.run_config[k]
    except Exception:
        pass
    return rc


def _safe_tag(s: str) -> str:
    s = str(s).strip()
    s = s.replace("/", "-").replace("\\", "-").replace(" ", "-")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "unknown"


def _alpha_tag(alpha: float) -> str:
    txt = f"{alpha:.6g}"
    return txt.replace(".", "p")


def _model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    sd = model.state_dict()
    total_bytes = 0
    named_shapes: Dict[str, Any] = {}

    for name, t in sd.items():
        if hasattr(t, "numel") and hasattr(t, "element_size"):
            total_bytes += int(t.numel() * t.element_size())
        named_shapes[name] = tuple(t.shape)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "state_dict_tensors": len(sd),
        "state_dict_bytes": int(total_bytes),
        "state_dict_mb": float(total_bytes) / (1024.0 * 1024.0),
        "named_shapes": named_shapes,
        "repr": repr(model),
    }


def _write_header(
    log_path: Path,
    *,
    dataset_name: str,
    data_dir: str,
    num_rounds: int,
    lr: float,
    batch_size: int,
    local_epochs: int,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    fraction_train: float,
    scheme: str,
    alpha: float,
    seed: int,
    min_partition_size: int,
    device: torch.device,
    model: torch.nn.Module,
    run_config_snapshot: Dict[str, Any],
    partition_plot_path: Optional[Path],
    final_model_path: Path,
    aggregation: str,
    aggregation_params: Dict[str, Any],
    experiment_dir: Path,
) -> None:
    ms = _model_summary(model)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tv_ver = getattr(torchvision, "__version__", "not-installed") if torchvision else "not-installed"

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Flower training log\n")
        f.write(f"Start time: {now}\n\n")

        f.write("== Environment ==\n")
        f.write(f"python: {sys.version.replace(chr(10), ' ')}\n")
        f.write(f"platform: {platform.platform()}\n")
        f.write(f"flwr: {flwr.__version__}\n")
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"torchvision: {tv_ver}\n")
        f.write(f"device (server eval): {str(device)}\n\n")

        f.write("== Experiment folder ==\n")
        f.write(f"experiment_dir: {str(experiment_dir)}\n\n")

        f.write("== Federated config (resolved) ==\n")
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"data_dir: {data_dir}\n")
        f.write(f"num_rounds: {num_rounds}\n")

        # гиперпараметры обучения (у клиентов)
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"local_epochs: {local_epochs}\n")
        f.write(f"learning_rate: {lr}\n")
        f.write(f"momentum: {momentum}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"num_workers: {num_workers}\n")

        f.write(f"fraction_train: {fraction_train}\n")
        f.write("fraction_evaluate: 0.0 (client-side eval disabled)\n")

        f.write(f"split_scheme: {scheme}\n")
        if scheme == "dirichlet":
            f.write(f"dirichlet_alpha: {alpha}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"min_partition_size: {min_partition_size}\n")

        f.write(f"aggregation: {aggregation}\n")
        if aggregation_params:
            f.write(f"aggregation_params: {aggregation_params}\n")

        f.write("weighted_by: num-examples\n\n")

        f.write("== Artifacts ==\n")
        f.write(f"log_file: {str(log_path)}\n")
        if partition_plot_path is None:
            f.write("partition_plot_file: (disabled)\n")
        else:
            f.write(f"partition_plot_file: {str(partition_plot_path)}\n")
            f.write(f"partition_plot_exists: {partition_plot_path.exists()}\n")
        f.write(f"final_model_file: {str(final_model_path)}\n\n")

        f.write("== Full run_config snapshot ==\n")
        for k in sorted(run_config_snapshot.keys()):
            f.write(f"{k}: {run_config_snapshot[k]}\n")
        f.write("\n")

        f.write("== Model summary ==\n")
        f.write(f"total_params: {ms['total_params']}\n")
        f.write(f"trainable_params: {ms['trainable_params']}\n")
        f.write(f"state_dict_tensors: {ms['state_dict_tensors']}\n")
        f.write(f"state_dict_size: {ms['state_dict_mb']:.3f} MB\n\n")

        f.write("Model repr:\n")
        f.write(ms["repr"])
        f.write("\n\n")

        f.write("State dict tensors (name -> shape):\n")
        for name, shp in ms["named_shapes"].items():
            f.write(f"  {name}: {shp}\n")
        f.write("\n")


def _make_logging_strategy_cls(base_cls: Type[Any]) -> Type[Any]:
    """Подкласс выбранной стратегии: перехватываем aggregate_train и сохраняем client details."""
    class LoggingStrategy(base_cls):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.round_client_logs: Dict[int, Dict[int, Dict[str, Any]]] = {}

        def aggregate_train(
            self, server_round: int, replies: Iterable[Message]
        ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
            replies_list = list(replies)

            per_round: Dict[int, Dict[str, Any]] = {}
            for rep in replies_list:
                src = rep.metadata.src_node_id
                if rep.content is None or "details" not in rep.content:
                    continue

                details: ConfigRecord = rep.content["details"]  # type: ignore[assignment]
                cid = int(details.get("partition-id", src))
                per_round[cid] = {
                    "epoch_losses": list(details.get("epoch-train-losses", [])),
                    "round_time_sec": float(details.get("round-time-sec", 0.0)),
                    "local_epochs": int(details.get("local-epochs", 0)),
                    "num_examples": int(details.get("num-examples", 0)),
                    "src_node_id": int(src),
                }

            self.round_client_logs[server_round] = per_round
            return super().aggregate_train(server_round, replies_list)

        def get_round_logs(self, server_round: int) -> Dict[int, Dict[str, Any]]:
            return self.round_client_logs.get(server_round, {})

    return LoggingStrategy


def _pick_strategy(context: Context, fraction_train: float) -> Tuple[Any, str, Dict[str, Any]]:
    agg = str(_rc(context, "aggregation", "agg", default="fedavg")).strip().lower()

    # Общие параметры (важно для robust-стратегий: чаще нужен минимум 2 клиента)
    min_train_nodes = int(_rc(context, "min-train-nodes", "min_train_nodes", default=1))
    min_available_nodes = int(_rc(context, "min-available-nodes", "min_available_nodes", default=1))

    common = dict(
        fraction_train=float(fraction_train),
        fraction_evaluate=0.0,  # client-side eval disabled
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=0,
        min_available_nodes=min_available_nodes,
        weighted_by_key="num-examples",
        arrayrecord_key="arrays",
        configrecord_key="config",
    )

    params: Dict[str, Any] = {}

    if agg in {"fedavg", "avg"}:
        base_cls = FedAvg

    elif agg in {"fedavgm", "avgm"}:
        base_cls = FedAvgM
        params = {
            "server_learning_rate": float(_rc(context, "server-learning-rate", "server_learning_rate", default=1.0)),
            "server_momentum": float(_rc(context, "server-momentum", "server_momentum", default=0.0)),
        }

    elif agg in {"fedadam", "adam"}:
        base_cls = FedAdam
        params = {
            "eta": float(_rc(context, "server-eta", "eta", default=0.01)),
            "eta_l": float(_rc(context, "server-eta-l", "eta_l", default=0.001)),
            "beta_1": float(_rc(context, "server-beta1", "beta_1", default=0.9)),
            "beta_2": float(_rc(context, "server-beta2", "beta_2", default=0.99)),
            "tau": float(_rc(context, "server-tau", "tau", default=1e-9)),
        }

    elif agg in {"fedyogi", "yogi"}:
        base_cls = FedYogi
        params = {
            "eta": float(_rc(context, "server-eta", "eta", default=0.01)),
            "eta_l": float(_rc(context, "server-eta-l", "eta_l", default=0.001)),
            "beta_1": float(_rc(context, "server-beta1", "beta_1", default=0.9)),
            "beta_2": float(_rc(context, "server-beta2", "beta_2", default=0.99)),
            "tau": float(_rc(context, "server-tau", "tau", default=1e-9)),
        }

    elif agg in {"fedadagrad", "adagrad"}:
        base_cls = FedAdagrad
        params = {
            "eta": float(_rc(context, "server-eta", "eta", default=0.01)),
            "eta_l": float(_rc(context, "server-eta-l", "eta_l", default=0.001)),
            "tau": float(_rc(context, "server-tau", "tau", default=1e-9)),
        }

    elif agg in {"fedmedian", "median"}:
        base_cls = FedMedian
        common["min_train_nodes"] = max(int(common["min_train_nodes"]), 2)
        common["min_available_nodes"] = max(int(common["min_available_nodes"]), 2)

    elif agg in {"fedtrimmedavg", "trimmed"}:
        base_cls = FedTrimmedAvg
        params = {"beta": float(_rc(context, "trimmed-beta", "beta", default=0.2))}
        common["min_train_nodes"] = max(int(common["min_train_nodes"]), 2)
        common["min_available_nodes"] = max(int(common["min_available_nodes"]), 2)

    elif agg in {"krum"}:
        base_cls = Krum
        params = {"num_malicious_nodes": int(_rc(context, "num-malicious-nodes", "num_malicious_nodes", default=0))}
        common["min_train_nodes"] = max(int(common["min_train_nodes"]), 2)
        common["min_available_nodes"] = max(int(common["min_available_nodes"]), 2)

    elif agg in {"multikrum"}:
        base_cls = MultiKrum
        params = {
            "num_malicious_nodes": int(_rc(context, "num-malicious-nodes", "num_malicious_nodes", default=0)),
            "num_nodes_to_select": int(_rc(context, "num-nodes-to-select", "num_nodes_to_select", default=1)),
        }
        common["min_train_nodes"] = max(int(common["min_train_nodes"]), 2)
        common["min_available_nodes"] = max(int(common["min_available_nodes"]), 2)

    elif agg in {"bulyan"}:
        base_cls = Bulyan
        params = {"num_malicious_nodes": int(_rc(context, "num-malicious-nodes", "num_malicious_nodes", default=0))}
        common["min_train_nodes"] = max(int(common["min_train_nodes"]), 2)
        common["min_available_nodes"] = max(int(common["min_available_nodes"]), 2)

    else:
        raise ValueError(
            f"Unknown aggregation='{agg}'. "
            "Use one of: fedavg, fedavgm, fedadam, fedyogi, fedadagrad, "
            "fedmedian, fedtrimmedavg, krum, multikrum, bulyan."
        )

    LoggingCls = _make_logging_strategy_cls(base_cls)
    strategy = LoggingCls(**common, **params)
    return strategy, agg, params


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # --- base config ---
    dataset_name: str = _rc(context, "dataset", "dataset_name", default="cifar10")
    data_dir: str = _rc(context, "data-dir", "data_dir", default="data/")

    scheme: str = _rc(context, "scheme", "split_scheme", default="iid")
    alpha: float = float(_rc(context, "alpha", default=0.3))
    seed: int = int(_rc(context, "seed", default=42))
    min_partition_size: int = int(_rc(context, "min-partition-size", "min_partition_size", default=0))

    num_rounds: int = int(_rc(context, "num-server-rounds", "num_rounds", "num-rounds", default=10))
    fraction_train: float = float(_rc(context, "fraction-train", "fraction_train", default=1.0))

    # число клиентов (для разбиения/визуализации/логов)
    num_clients_for_partition = int(_rc(context, "num-clients", "num_clients", default=10))

    # --- defaults (task.py) ---
    hp = default_hparams(dataset_name)
    lr: float = float(_rc(context, "learning-rate", "lr", default=hp.lr))
    batch_size: int = int(_rc(context, "batch-size", "batch_size", default=hp.batch_size))
    local_epochs: int = int(_rc(context, "local-epochs", "local_epochs", default=hp.local_epochs))
    momentum: float = float(_rc(context, "momentum", default=hp.momentum))
    weight_decay: float = float(_rc(context, "weight_decay", default=hp.weight_decay))
    num_workers: int = int(_rc(context, "num-workers", "num_workers", default=hp.num_workers))

    # --- choose aggregation strategy ---
    strategy, agg_name, agg_params = _pick_strategy(context, fraction_train=fraction_train)

    # --- experiment folder name: dataset + scheme (+alpha if dirichlet) + aggregation ---
    ds_tag = _safe_tag(dataset_name)
    sc_tag = _safe_tag(scheme)
    ag_tag = _safe_tag(agg_name)

    exp_name = f"{ds_tag}__{sc_tag}"
    if scheme == "dirichlet":
        exp_name += f"__a{_alpha_tag(alpha)}"
    exp_name += f"__{ag_tag}"

    # (файлы внутри делаем уникальнее за счёт num-clients)
    run_base = f"{exp_name}__n{int(num_clients_for_partition)}"

    # куда складывать эксперименты
    runs_dir = Path(_rc(context, "runs-dir", "runs_dir", default="runs"))
    experiment_dir = runs_dir / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_path = experiment_dir / f"{run_base}.log"
    partition_plot_path = experiment_dir / f"{run_base}__partition.png"
    final_model_path = experiment_dir / f"{run_base}.pt"

    # ---- model ----
    global_model = create_model(dataset_name)
    initial_arrays = ArrayRecord(global_model.state_dict())

    # ---- partition visualization + server test loader ----
    fed = prepare_federated_dataset(
        dataset=dataset_name,
        num_clients=num_clients_for_partition,
        scheme=scheme,
        alpha=alpha,
        seed=seed,
        data_dir=data_dir,
        split_train="train",
        split_test="test",
        min_partition_size=min_partition_size,
        save_plot_to=str(partition_plot_path),
        plot_type=_rc(context, "partition-plot-type", default="heatmap"),
        size_unit=_rc(context, "partition-size-unit", default="percent"),
    )

    testloader, meta = make_server_testloader(
        fed=fed,
        test_transform=None,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    device = get_device(prefer_cuda=True)

    # ---- write header FIRST ----
    run_config_snapshot = _run_config_to_dict(context)
    _write_header(
        log_path,
        dataset_name=dataset_name,
        data_dir=data_dir,
        num_rounds=num_rounds,
        lr=lr,
        batch_size=batch_size,
        local_epochs=local_epochs,
        momentum=momentum,
        weight_decay=weight_decay,
        num_workers=num_workers,
        fraction_train=fraction_train,
        scheme=scheme,
        alpha=alpha,
        seed=seed,
        min_partition_size=min_partition_size,
        device=device,
        model=global_model,
        run_config_snapshot=run_config_snapshot,
        partition_plot_path=partition_plot_path,
        final_model_path=final_model_path,
        aggregation=agg_name,
        aggregation_params=agg_params,
        experiment_dir=experiment_dir,
    )

    def append_round_log(server_round: int, server_metrics: MetricRecord) -> None:
        client_logs = strategy.get_round_logs(server_round)  # type: ignore[attr-defined]

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"Round {server_round}:\n")

            for cid in sorted(client_logs.keys()):
                payload = client_logs[cid]
                epoch_losses = payload.get("epoch_losses", [])
                time_sec = float(payload.get("round_time_sec", 0.0))

                f.write(f"    Client{cid + 1}:\n")
                for i, loss in enumerate(epoch_losses):
                    f.write(f"        loc_epoch {i}: train loss - {float(loss):.6f}\n")
                f.write(f"        time on round {server_round} - {time_sec:.2f}s\n")

            test_acc = float(server_metrics.get("test_acc", 0.0))
            test_loss = float(server_metrics.get("test_loss", 0.0))
            f.write(f"    Server_eval: test_acc: {test_acc:.4f} | test_loss: {test_loss:.6f}\n\n")

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = create_model(dataset_name)
        model.load_state_dict(arrays.to_torch_state_dict())

        loss, acc = evaluate_global(
            model=model,
            testloader=testloader,
            device=device,
            img_col=meta["img_col"],
            label_col=meta["label_col"],
        )

        metrics = MetricRecord(
            {
                "test_loss": float(loss),
                "test_acc": float(acc),
                "num-examples": int(len(testloader.dataset)),
                "round": int(server_round),
            }
        )

        append_round_log(server_round, metrics)
        return metrics

    # ---- start ----
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"learning-rate": lr}),
        evaluate_fn=global_evaluate,
    )

    torch.save(result.arrays.to_torch_state_dict(), str(final_model_path))
    print("\nSaved final model to:", final_model_path.resolve())
    print("Saved log to:", log_path.resolve())
    print("Saved partition plot to:", partition_plot_path.resolve())
    print("Experiment folder:", experiment_dir.resolve())
